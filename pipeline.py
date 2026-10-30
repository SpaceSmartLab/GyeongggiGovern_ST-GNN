"""Reproducible temperature/RH forecasting. Read ZIP without modifying source.

Run: python pipeline.py --stage all
All timestamps remain naive local timestamps; logger timezone is not verified.
"""
import os
from pathlib import Path

ROOT = Path(__file__).absolute().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
import copy
import hashlib
import io
import json
import random
import re
import sys
import time
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VARIABLES = ["temperature_c", "relative_humidity_pct"]
# Approximate normalized coordinates manually read from supplied plans, not metres.
# North and south labels follow CSV labels, not an inferred geographic bearing.
NODES = [
    ("20896478", "S04", .680, .625),
    ("20896479", "S05", .277, .625),
    ("20896481", "S07", .516, .625),
    ("21864661", "S11", .516, .625),
    ("20867863", "S18", None, None),
    ("21221269", "S31", .817, .625),
    ("21221256", "S37", .192, .625),
    ("21221267", "S41", .369, .625),
    ("20846624", "N01", .378, .444),
    ("20867861", "N02", .478, .483),
    ("20867860", "N03", .296, .445),
    ("20846631", "N05", .680, .444),
    ("20846620", "N15", .258, .483),
    ("20846621", "N25", .773, .480),
]

def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

def read_config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

def make_graph(k=3):
    n = len(NODES)
    graph = np.eye(n, dtype=np.float32)
    xy = np.array([[x if x is not None else np.nan, y if y is not None else np.nan]
                   for _, _, x, y in NODES])
    distances = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    for i in range(n):
        if np.isnan(xy[i]).any():
            continue
        order = [j for j in np.argsort(distances[i]) if j != i and np.isfinite(distances[i, j])][:k]
        for j in order:
            w = float(np.exp(-(distances[i, j] / .20) ** 2))
            graph[i, j] = graph[j, i] = w
    return graph

def deduplicate(frame):
    # Prefer collection v2; keep the final source row for within-file conflicts.
    return (frame.sort_values(["serial", "timestamp", "collection", "source_file", "source_row"])
            .drop_duplicates(["serial", "timestamp"], keep="last"))

def causal_fill(frame, limit=2):
    return frame.ffill(limit=limit)

class DirectorySource:
    def __init__(self, root):
        self.root = root
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def namelist(self):
        return [str(p.relative_to(self.root)).replace("\\", "/") for p in self.root.rglob("*") if p.is_file()]
    def read(self, name):
        return (self.root / name).read_bytes()

def preprocess(cfg):
    out = ROOT / "data" / "processed"
    audit = ROOT / "outputs" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    all_frames, inventory = [], []
    source = Path(cfg["source_path"])
    with (DirectorySource(source) if source.is_dir() else zipfile.ZipFile(source)) as z:
        for name in sorted(z.namelist()):
            if not name.lower().endswith(".csv"):
                continue
            payload = z.read(name)
            try:
                text = payload.decode("utf-8-sig")
                encoding = "utf-8-sig"
            except UnicodeDecodeError:
                text = payload.decode("cp949")
                encoding = "cp949"
            raw = pd.read_csv(io.StringIO(text))
            serial = re.search(r"\((\d+)\)", name).group(1)
            collection = 2 if "2차" in name else 1
            frame = pd.DataFrame({
                "timestamp": pd.to_datetime(raw.iloc[:, 0], format="mixed", errors="coerce"),
                "serial": serial,
                "temperature_c": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
                "relative_humidity_pct": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
                "collection": collection,
                "source_file": name,
                "source_row": np.arange(len(raw)) + 2,
            })
            bad_time = frame.timestamp.isna()
            if bad_time.any():
                frame.loc[bad_time].to_csv(audit / f"invalid_timestamp_{collection}_{serial}.csv", index=False, encoding="utf-8-sig")
            inventory.append({"source_file": name, "serial": serial, "collection": collection,
                              "encoding": encoding, "sha256": hashlib.sha256(payload).hexdigest(),
                              "rows": len(frame), "invalid_timestamp_rows": int(bad_time.sum()),
                              "first": frame.timestamp.min(), "last": frame.timestamp.max()})
            all_frames.append(frame.loc[~bad_time])
    pd.DataFrame(inventory).to_csv(audit / "file_inventory.csv", index=False, encoding="utf-8-sig")
    raw_all = pd.concat(all_frames, ignore_index=True)
    raw_all.to_csv(out / "source_observations.csv.gz", index=False, compression="gzip")
    duplicated = raw_all.duplicated(["serial", "timestamp"], keep=False)
    duplicates = raw_all.loc[duplicated]
    spans = duplicates.groupby(["serial", "timestamp"])[VARIABLES].agg(lambda s: s.max() - s.min())
    conflict = spans[(spans.temperature_c > .011) | (spans.relative_humidity_pct > .011)]
    conflict.to_csv(audit / "duplicate_value_conflicts.csv", encoding="utf-8-sig")
    merged = deduplicate(raw_all).copy()
    invalid_counts = {}
    for col, lower, upper in [(VARIABLES[0], -20., 60.), (VARIABLES[1], 0., 100.)]:
        invalid = merged[col].notna() & ~merged[col].between(lower, upper)
        invalid_counts[col] = int(invalid.sum())
        if invalid.any():
            merged.loc[invalid].to_csv(audit / f"invalid_{col}.csv", index=False, encoding="utf-8-sig")
        merged.loc[invalid, col] = np.nan
    exact = merged.timestamp.eq(merged.timestamp.dt.floor("10min"))
    merged.loc[~exact].to_csv(audit / "off_grid_records.csv", index=False, encoding="utf-8-sig")
    # Off-grid export/event rows are preserved above and excluded, never shifted in time.
    clean = merged.loc[exact].copy()
    clean.to_csv(out / "clean_observations_all_dates.csv.gz", index=False, compression="gzip")
    times = pd.date_range(clean.timestamp.min(), pd.Timestamp(cfg["analysis_end"]), freq="10min")
    serials = [x[0] for x in NODES]
    y = np.full((len(times), len(serials), 2), np.nan, dtype=np.float32)
    coverage = []
    for j, (serial, label, x, yy) in enumerate(NODES):
        records = clean.loc[clean.serial.eq(serial)].set_index("timestamp").sort_index()
        aligned = records[VARIABLES].reindex(times)
        y[:, j] = aligned.to_numpy(np.float32)
        coverage.append({"serial": serial, "label": label, "x_normalized": x, "y_normalized": yy,
                         "location_status": "unverified_self_only" if x is None else "approximate_plan_reading",
                         "first_record": records.index.min(), "last_record": records.index.max(),
                         "observed_points_primary": int(aligned.notna().all(axis=1).sum()),
                         "temperature_min": records.temperature_c.min(), "temperature_max": records.temperature_c.max(),
                         "rh_min": records.relative_humidity_pct.min(), "rh_max": records.relative_humidity_pct.max()})
    pd.DataFrame(coverage).to_csv(out / "sensor_metadata.csv", index=False, encoding="utf-8-sig")
    observed = np.isfinite(y)
    filled = np.empty_like(y)
    age = np.zeros_like(y)
    for j in range(len(serials)):
        filled[:, j] = causal_fill(pd.DataFrame(y[:, j]), cfg["forward_fill_limit"]).to_numpy()
        for v in range(2):
            elapsed = cfg["history_steps"]
            for t in range(len(times)):
                elapsed = 0 if observed[t, j, v] else elapsed + 1
                age[t, j, v] = min(elapsed, cfg["history_steps"]) / cfg["history_steps"]
    available = np.isfinite(filled)
    train_time = times < pd.Timestamp(cfg["validation_start"])
    mean = np.nanmean(y[train_time], axis=(0, 1)).astype(np.float32)
    std = np.nanstd(y[train_time], axis=(0, 1)).astype(np.float32)
    if (std < 1e-6).any():
        raise ValueError("Near-constant training target")
    scaled = np.nan_to_num((filled - mean) / std, nan=0.)
    minute = times.hour.to_numpy() * 60 + times.minute.to_numpy()
    week = times.dayofweek.to_numpy() + minute / 1440.
    calendar = np.stack([np.sin(2*np.pi*minute/1440), np.cos(2*np.pi*minute/1440),
                         np.sin(2*np.pi*week/7), np.cos(2*np.pi*week/7),
                         (times.dayofweek.to_numpy() < 5).astype(float)], axis=-1).astype(np.float32)
    features = np.concatenate([scaled, observed.astype(np.float32), available.astype(np.float32),
                               age, np.broadcast_to(calendar[:, None], (len(times), len(serials), 5))], axis=-1)
    graph = make_graph(cfg["graph_k"])
    np.savez_compressed(out / "dataset.npz", times=times.to_numpy().astype("datetime64[ns]"),
                        y=y, filled=filled, observed=observed, available=available,
                        x=features, mean=mean, std=std, graph=graph,
                        serials=np.array(serials), labels=np.array([x[1] for x in NODES]))
    pd.DataFrame(graph, index=[x[1] for x in NODES], columns=[x[1] for x in NODES]).to_csv(out / "adjacency.csv")
    flat = pd.DataFrame({"timestamp": np.repeat(times, len(serials)), "serial": np.tile(serials, len(times))})
    for v, col in enumerate(VARIABLES):
        flat[col] = y[..., v].reshape(-1)
        flat[col + "_input"] = filled[..., v].reshape(-1)
        flat[col + "_observed"] = observed[..., v].reshape(-1)
    flat.to_csv(out / "aligned_10min.csv.gz", index=False, compression="gzip")
    tail = clean.loc[clean.timestamp.gt(pd.Timestamp(cfg["analysis_end"]))]
    tail.to_csv(out / "post_primary_period.csv.gz", index=False, compression="gzip")
    # Flat runs are an audit flag, not automatically an error.
    flat_runs = []
    for j, node in enumerate(NODES):
        for v, col in enumerate(VARIABLES):
            values = y[:, j, v]
            longest, run = 0, 0
            for t, val in enumerate(values):
                run = run + 1 if t and np.isfinite(val) and val == values[t-1] else int(np.isfinite(val))
                longest = max(longest, run)
            flat_runs.append({"label": node[1], "variable": col, "longest_exact_constant_steps": longest})
    pd.DataFrame(flat_runs).to_csv(audit / "constant_runs.csv", index=False)
    summary = {"source_path": str(source), "source_kind": "directory" if source.is_dir() else "zip",
               "source_manifest_sha256": hashlib.sha256(json.dumps(inventory, default=str, sort_keys=True).encode()).hexdigest(),
               "csv_files": len(inventory), "unique_devices": len(NODES),
               "valid_timestamp_source_rows": len(raw_all), "duplicate_rows_removed": len(raw_all)-len(merged),
               "duplicate_groups": len(spans), "conflict_groups_above_0_011": len(conflict),
               "invalid_values": invalid_counts, "off_grid_records_excluded": int((~exact).sum()),
               "primary_start": times[0], "primary_end": times[-1], "primary_steps": len(times),
               "post_primary_rows_retained": len(tail),
               "observed_target_values": int(observed.sum()),
               "input_values_forward_filled": int((available & ~observed).sum()),
               "scaler_mean_train_only": mean.tolist(), "scaler_std_train_only": std.tolist(),
               "notes": ["Raw source is unchanged; collection 2 takes precedence in duplicate groups.",
                         "Logger local timezone and exact installation dates remain unverified.",
                         "S18 has no verified coordinate and only a self edge.",
                         "Missing targets are masked, not imputed. Only inputs are forward filled, at most 20 minutes.",
                         "No survey, occupancy snapshot, CO2 or dew point used as predictors."]}
    dump_json(audit / "preprocessing_summary.json", summary)
    print("PREPROCESS", json.dumps(summary, ensure_ascii=False, default=str), flush=True)

def load_data():
    with np.load(ROOT / "data" / "processed" / "dataset.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def split_origins(data, cfg):
    ts = pd.DatetimeIndex(data["times"])
    h = np.array(cfg["horizon_steps"])
    result = {}
    bounds = {"train": (ts[0], pd.Timestamp(cfg["validation_start"])),
              "validation": (pd.Timestamp(cfg["validation_start"]), pd.Timestamp(cfg["test_start"])),
              "test": (pd.Timestamp(cfg["test_start"]), pd.Timestamp(cfg["analysis_end"]) + pd.Timedelta(minutes=10))}
    for name, (start, end) in bounds.items():
        origins = []
        for i in range(cfg["history_steps"]-1, len(ts)-max(h)):
            # Origins and all targets must be in their partition. Past context is allowed.
            if not (ts[i] >= start and ts[i+max(h)] < end):
                continue
            mask = data["observed"][i+h] & data["available"][i][None]
            if mask.any():
                origins.append(i)
        result[name] = np.array(origins, dtype=np.int64)
    return result

class Windows(Dataset):
    def __init__(self, data, origins, cfg):
        self.d, self.origins, self.cfg = data, origins, cfg
    def __len__(self):
        return len(self.origins)
    def __getitem__(self, index):
        i = self.origins[index]
        h = np.array(self.cfg["horizon_steps"])
        x = self.d["x"][i-self.cfg["history_steps"]+1:i+1].transpose(2, 0, 1).copy()
        y = (self.d["y"][i+h] - self.d["mean"]) / self.d["std"]
        mask = self.d["observed"][i+h] & self.d["available"][i][None]
        return torch.from_numpy(x), torch.from_numpy(np.nan_to_num(y)), torch.from_numpy(mask)

class STBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilation = dilation
        self.temporal = nn.Conv2d(channels, channels, (3, 1), dilation=(dilation, 1))
        self.spatial = nn.Conv2d(channels, channels, 1)
    def forward(self, z, graph, active):
        hidden = self.temporal(nn.functional.pad(z, (0, 0, 2*self.dilation, 0)))
        hidden = torch.relu(hidden)
        weights = graph[None, None] * active[:, :, None, :]  # remove unavailable source nodes
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        messages = torch.einsum("btij,bctj->bcti", weights, hidden)
        return torch.relu(z + self.spatial(messages)) * active[:, None]

class STGNN(nn.Module):
    def __init__(self, features, graph, hidden=24, horizons=3):
        super().__init__()
        self.register_buffer("graph", torch.as_tensor(graph, dtype=torch.float32))
        self.input_projection = nn.Conv2d(features, hidden, 1)
        self.blocks = nn.ModuleList([STBlock(hidden, d) for d in [1, 2, 4, 8]])
        self.head = nn.Conv2d(hidden, horizons*2, 1)
        self.horizons = horizons
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self, x):
        active = x[:, 4:6].amin(dim=1)
        z = torch.relu(self.input_projection(x)) * active[:, None]
        for block in self.blocks:
            z = block(z, self.graph, active)
        delta = self.head(z[:, :, -1:, :]).squeeze(2)
        delta = delta.reshape(x.shape[0], self.horizons, 2, x.shape[-1]).permute(0, 1, 3, 2)
        # Direct multi-horizon residual forecasts around the latest available observation.
        return delta + x[:, :2, -1, :].transpose(1, 2)[:, None]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def masked_loss(pred, target, mask):
    error = (pred-target).square()
    return (error*mask).sum() / mask.sum().clamp_min(1)

def predict_network(model, data, origins, cfg):
    model.eval()
    predictions = []
    with torch.no_grad():
        # Prediction must never require future labels, including at the final timestamp.
        for start in range(0, len(origins), cfg["batch_size"]):
            batch = origins[start:start+cfg["batch_size"]]
            x = torch.from_numpy(np.stack([data["x"][i-cfg["history_steps"]+1:i+1].transpose(2,0,1)
                                          for i in batch]).copy())
            predictions.append(model(x).numpy())
    return np.concatenate(predictions) * data["std"] + data["mean"]

def target_arrays(data, origins, cfg):
    positions = origins[:, None] + np.array(cfg["horizon_steps"])[None]
    target = data["y"][positions]
    mask = data["observed"][positions] & data["available"][origins, None]
    return target, mask

def score(pred, target, mask, std):
    error = np.where(mask, np.abs(pred-target), np.nan)
    return float(np.nanmean(error/std))

def train_network(name, graph, seed, data, splits, cfg):
    set_seed(seed)
    model = STGNN(data["x"].shape[-1], graph, cfg["hidden_channels"], len(cfg["horizon_steps"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    loader = DataLoader(Windows(data, splits["train"], cfg), batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    target, mask = target_arrays(data, splits["validation"], cfg)
    best, stale, history, best_epoch = float("inf"), 0, [], 0
    start = time.time()
    best_state = None
    for epoch in range(1, cfg["max_epochs"]+1):
        model.train()
        total, count = 0., 0
        for x, y, valid in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = masked_loss(pred, y, valid)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * valid.sum().item()
            count += valid.sum().item()
        val_pred = predict_network(model, data, splits["validation"], cfg)
        val = score(val_pred, target, mask, data["std"])
        history.append({"epoch": epoch, "train_scaled_mse": total/count, "validation_scaled_mae": val})
        if val < best - 1e-5:
            best, best_epoch, stale = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"TRAIN {name} seed={seed} epoch={epoch} train={total/count:.5f} val={val:.5f} best={best:.5f} elapsed={time.time()-start:.0f}s", flush=True)
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    destination = ROOT / "models" / f"{name}_seed{seed}.pt"
    destination.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "seed": seed, "name": name,
                "validation_score": best, "best_epoch": best_epoch,
                "features": data["x"].shape[-1], "mean": data["mean"].tolist(),
                "std": data["std"].tolist(), "serials": data["serials"].tolist()}, destination)
    history_dir = ROOT / "outputs" / "training"
    history_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_dir / f"{name}_seed{seed}.csv", index=False)
    metadata = {"name": name, "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
                "validation_scaled_mae": best, "parameters": sum(p.numel() for p in model.parameters()),
                "elapsed_seconds": time.time()-start, "checkpoint": str(destination.relative_to(ROOT))}
    print("FINISHED", json.dumps(metadata), flush=True)
    return model, metadata

def ridge_features(data, origins, cfg, spatial=False):
    # Shared per-node linear model. Own lag values/masks + current calendar information.
    lags = origins[:, None] - np.arange(cfg["history_steps"]-1, -1, -1)[None]
    local = data["x"][lags, :, :6].transpose(0, 2, 1, 3)
    features = [local.reshape(len(origins), len(NODES), -1), data["x"][origins, :, 8:]]
    if spatial:
        avail = data["available"][lags].all(-1).astype(np.float32)
        weights = data["graph"][None, None] * avail[:, :, None, :]
        weights = weights / np.maximum(weights.sum(-1, keepdims=True), 1e-8)
        neighbors = np.einsum("btij,btjf->btif", weights, data["x"][lags, :, :2])
        features.append(neighbors.transpose(0, 2, 1, 3).reshape(len(origins), len(NODES), -1))
    x = np.concatenate(features, axis=-1).astype(np.float64)
    return np.concatenate([x, np.ones((*x.shape[:2], 1))], axis=-1)

def ridge_train_predict(data, splits, cfg, spatial=False):
    train_x = ridge_features(data, splits["train"], cfg, spatial)
    val_x = ridge_features(data, splits["validation"], cfg, spatial)
    train_y, train_m = target_arrays(data, splits["train"], cfg)
    val_y, val_m = target_arrays(data, splits["validation"], cfg)
    base = data["filled"][splits["train"]]
    delta_y = (train_y-base[:, None]) / data["std"]
    best, selected, best_weights = float("inf"), None, None
    gram = []
    for hi in range(len(cfg["horizon_steps"])):
        row = []
        for v in range(2):
            valid = train_m[:, hi, :, v]
            xx, yy = train_x[valid], delta_y[:, hi, :, v][valid]
            row.append((xx.T @ xx, xx.T @ yy))
        gram.append(row)
    for alpha in cfg["ridge_alphas"]:
        weights = np.empty((len(cfg["horizon_steps"]), 2, train_x.shape[-1]))
        penalty = np.eye(train_x.shape[-1]) * alpha
        penalty[-1, -1] = 0
        for hi in range(len(cfg["horizon_steps"])):
            for v in range(2):
                xtx, xty = gram[hi][v]
                weights[hi, v] = np.linalg.solve(xtx+penalty, xty)
        pred = np.einsum("onf,hvf->ohnv", val_x, weights)*data["std"] + data["filled"][splits["validation"], None]
        value = score(pred, val_y, val_m, data["std"])
        if value < best:
            best, selected, best_weights = value, alpha, weights.copy()
    name = "ridge_spatial" if spatial else "ridge_temporal"
    np.savez(ROOT / "models" / f"{name}.npz", weights=best_weights, alpha=selected)
    return best_weights, {"name": name, "alpha": selected, "validation_scaled_mae": best}

def metrics_rows(name, pred, target, mask, data, origins, cfg, split="test"):
    rows = []
    ts = pd.DatetimeIndex(data["times"])
    for hi, horizon in enumerate(cfg["horizon_steps"]):
        future = ts[origins+horizon]
        # This is a weekday/time proxy, not a verified occupancy or holiday schedule.
        work = (future.dayofweek < 5) & (future.hour >= 9) & (future.hour < 18)
        for v, variable in enumerate(VARIABLES):
            for period, keep in [("all", np.ones(len(origins), bool)), ("weekday_09_18", work)]:
                for node in [-1] + list(range(len(NODES))):
                    m = mask[:, hi, :, v] & keep[:, None]
                    if node >= 0:
                        m = m & (np.arange(len(NODES))[None] == node)
                    err = (pred[:, hi, :, v]-target[:, hi, :, v])[m]
                    if not len(err):
                        continue
                    rows.append({"split": split, "model": name, "horizon_minutes": horizon*10,
                                 "variable": variable, "period": period,
                                 "sensor": "ALL" if node < 0 else NODES[node][1], "n": len(err),
                                 "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err**2))),
                                 "bias": float(np.mean(err))})
    return rows

def fit(cfg):
    torch.set_num_threads(cfg["torch_threads"])
    data, rows, records, models = load_data(), [], [], {}
    splits = split_origins(data, cfg)
    (ROOT / "models").mkdir(exist_ok=True)
    dump_json(ROOT / "outputs" / "split_summary.json", {
        name: {"windows": len(idx), "first_origin": str(data["times"][idx[0]]),
               "last_origin": str(data["times"][idx[-1]]),
               "last_target": str(data["times"][idx[-1]+max(cfg["horizon_steps"])])}
        for name, idx in splits.items()})
    print("SPLITS", {k: len(v) for k, v in splits.items()}, flush=True)
    for name, graph in [("temporal_tcn", np.eye(len(NODES), dtype=np.float32)), ("stgnn", data["graph"])]:
        for seed in cfg["seeds"]:
            model, metadata = train_network(name, graph, seed, data, splits, cfg)
            models[f"{name}_seed{seed}"] = model
            records.append(metadata)
    for spatial in [False, True]:
        weights, metadata = ridge_train_predict(data, splits, cfg, spatial)
        models[metadata["name"]] = weights
        records.append(metadata)
    chosen = {}
    for name in ["temporal_tcn", "stgnn"]:
        selected = min([r for r in records if r["name"] == name], key=lambda r: r["validation_scaled_mae"])
        chosen[name] = f"{name}_seed{selected['seed']}"
    # Selection is frozen before any test predictions or metrics are computed.
    dump_json(ROOT / "outputs" / "model_selection.json", {"selected_by_validation_only": chosen, "runs": records})
    predictions = {}
    for split in ["validation", "test"]:
        origins = splits[split]
        target, mask = target_arrays(data, origins, cfg)
        predset = {"persistence": np.repeat(data["filled"][origins, None], len(cfg["horizon_steps"]), axis=1)}
        positions = origins[:, None]+np.array(cfg["horizon_steps"])[None]-144
        seasonal = data["filled"][np.maximum(positions, 0)].copy()
        # Missing previous-day values use persistence, identically for all comparisons.
        seasonal = np.where(np.isfinite(seasonal) & (positions >= 0)[:, :, None, None], seasonal, predset["persistence"])
        predset["previous_day_with_fallback"] = seasonal
        for name, model in models.items():
            if name.startswith("ridge"):
                x = ridge_features(data, origins, cfg, name == "ridge_spatial")
                predset[name] = np.einsum("onf,hvf->ohnv", x, model)*data["std"] + data["filled"][origins, None]
            else:
                predset[name] = predict_network(model, data, origins, cfg)
        for name, selected in chosen.items():
            predset[name] = predset[selected]
        for name, pred in predset.items():
            if not np.isfinite(pred[mask]).all():
                raise ValueError(f"Non-finite predictions in evaluation: {name}")
            rows.extend(metrics_rows(name, pred, target, mask, data, origins, cfg, split))
        if split == "test":
            predictions = predset
            np.savez_compressed(ROOT / "outputs" / "test_predictions.npz", origins=origins, target=target,
                                mask=mask, **predset)
    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "metrics.csv", index=False)
    # Compact, human-readable long forecasts for the selected ST-GNN and baselines.
    origins = splits["test"]
    target, mask = target_arrays(data, origins, cfg)
    records_long = []
    for hi, h in enumerate(cfg["horizon_steps"]):
        oi, ni, vi = np.where(mask[:, hi])
        frame = pd.DataFrame({"origin": data["times"][origins[oi]], "target_time": data["times"][origins[oi]+h],
                              "horizon_minutes": h*10, "sensor": data["labels"][ni], "serial": data["serials"][ni],
                              "variable": np.array(VARIABLES)[vi], "actual": target[oi, hi, ni, vi]})
        for name in ["persistence", "ridge_temporal", "ridge_spatial", "temporal_tcn", "stgnn"]:
            frame[name] = predictions[name][oi, hi, ni, vi]
        records_long.append(frame)
    pd.concat(records_long).to_csv(ROOT / "outputs" / "test_forecasts.csv.gz", index=False, compression="gzip")
    bootstrap = []
    rng = np.random.default_rng(20261012)
    dates = pd.DatetimeIndex(data["times"][origins]).normalize()
    unique_days = dates.unique()
    for baseline in ["persistence", "temporal_tcn", "ridge_spatial"]:
        for hi, h in enumerate(cfg["horizon_steps"]):
            for v, variable in enumerate(VARIABLES):
                diff = np.abs(predictions["stgnn"][:, hi, :, v]-target[:, hi, :, v]) - np.abs(predictions[baseline][:, hi, :, v]-target[:, hi, :, v])
                valid = mask[:, hi, :, v]
                sums, counts = [], []
                for date in unique_days:
                    selected = (dates == date)[:, None] & valid
                    sums.append(float(diff[selected].sum()))
                    counts.append(int(selected.sum()))
                sample = rng.integers(len(unique_days), size=(2000, len(unique_days)))
                bs = np.asarray(sums)[sample].sum(1) / np.asarray(counts)[sample].sum(1)
                bootstrap.append({"baseline": baseline, "variable": variable, "horizon_minutes": h*10,
                                  "stgnn_minus_baseline_mae": sum(sums)/sum(counts),
                                  "ci95_low": float(np.quantile(bs, .025)), "ci95_high": float(np.quantile(bs, .975)),
                                  "origin_day_blocks": len(unique_days)})
    pd.DataFrame(bootstrap).to_csv(ROOT / "outputs" / "paired_day_bootstrap.csv", index=False)
    # Retrospective issuance at the primary cutoff: future measurements are NOT used here.
    deployment = models[chosen["stgnn"]]
    final_origin = len(data["times"])-1
    forecast = predict_network(deployment, data, np.array([final_origin]), cfg)[0]
    future_rows = []
    for hi, h in enumerate(cfg["horizon_steps"]):
        for ni, node in enumerate(NODES):
            if data["available"][final_origin, ni].all():
                future_rows.append({"issued_at": str(data["times"][final_origin]),
                                    "forecast_time": str(pd.Timestamp(data["times"][final_origin])+pd.Timedelta(minutes=10*h)),
                                    "sensor": node[1], "serial": node[0],
                                    "temperature_c": forecast[hi, ni, 0], "relative_humidity_pct": forecast[hi, ni, 1],
                                    "status": "historical_cutoff_forecast_not_live"})
    pd.DataFrame(future_rows).to_csv(ROOT / "outputs" / "cutoff_forecast.csv", index=False)
    print("TRAINING_AND_TEST_COMPLETE", chosen, flush=True)

def plot_results(cfg):
    import matplotlib.pyplot as plt
    out = ROOT / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = load_data()
    metrics = pd.read_csv(ROOT / "outputs" / "metrics.csv")
    selected = ["persistence", "ridge_temporal", "ridge_spatial", "temporal_tcn", "stgnn"]
    m = metrics.query("split == 'test' and sensor == 'ALL' and period == 'all'")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for v, ax in enumerate(axs):
        for name in selected:
            rows = m[(m.model == name) & (m.variable == VARIABLES[v])]
            ax.plot(rows.horizon_minutes, rows.mae, marker="o", label=name)
        ax.set(xlabel="Forecast horizon (minutes)", ylabel="MAE (C)" if v == 0 else "MAE (percentage points)",
               title="Temperature" if v == 0 else "Relative humidity", xticks=[10,30,60])
        ax.grid(alpha=.25)
    axs[1].legend(fontsize=8)
    fig.suptitle("Held-out test: 14-21 September 2026")
    fig.savefig(out / "test_mae.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.imshow(data["observed"].all(-1).T, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0,vmax=1)
    ticks = np.linspace(0, len(data["times"])-1, 8).astype(int)
    ax.set(yticks=np.arange(len(NODES)), yticklabels=data["labels"], xticks=ticks,
           xticklabels=[str(data["times"][t])[:10] for t in ticks], title="Observed temperature and humidity (blue); missing or inactive (white)")
    for date, color in [(cfg["validation_start"],"orange"),(cfg["test_start"],"red")]:
        idx = np.searchsorted(data["times"], np.datetime64(date))
        ax.axvline(idx, color=color, linestyle="--")
    fig.savefig(out / "coverage.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    coords = np.array([[x or np.nan, y or np.nan] for _,_,x,y in NODES])
    for i in range(len(NODES)):
        for j in range(i):
            if data["graph"][i,j] > 0 and np.isfinite(coords[[i,j]]).all():
                ax.plot(coords[[i,j],0], coords[[i,j],1], color="gray", alpha=.4, linewidth=1)
    for i, node in enumerate(NODES):
        if np.isfinite(coords[i]).all():
            ax.scatter(*coords[i], color="steelblue" if node[1].startswith("S") else "darkorange", s=90)
            ax.annotate(node[1], coords[i], xytext=(0, 12 if node[1] != "S07" else -18), textcoords="offset points", ha="center")
    ax.invert_yaxis()
    ax.set_ylim(.665, .410)
    ax.set(xlabel="Normalized plan x (not metres)", ylabel="Normalized plan y", title="Approximate proximity graph; S18 self-only; availability gates all edges")
    fig.savefig(out / "graph.png", dpi=160)
    plt.close(fig)
    with np.load(ROOT / "outputs" / "test_predictions.npz") as z:
        origins, target, mask, pred, base = [z[k] for k in ["origins","target","mask","stgnn","persistence"]]
    ni = list(data["labels"]).index("S31")
    hi = 2
    times = pd.DatetimeIndex(data["times"][origins+cfg["horizon_steps"][hi]])
    keep = (times >= "2026-09-15") & (times < "2026-09-17")
    fig, axs = plt.subplots(2,1,figsize=(12,6),sharex=True,constrained_layout=True)
    for v, ax in enumerate(axs):
        ax.plot(times[keep], target[keep,hi,ni,v], label="Observed", color="black", linewidth=1.2)
        ax.plot(times[keep], pred[keep,hi,ni,v], label="ST-GNN", alpha=.85)
        ax.plot(times[keep], base[keep,hi,ni,v], label="Persistence", alpha=.65)
        ax.set_ylabel("Temperature (C)" if v == 0 else "RH (%)")
        ax.grid(alpha=.2)
    axs[0].legend(ncol=3)
    axs[0].set_title("S31: 60-minute-ahead forecasts, fixed illustrative test dates")
    fig.savefig(out / "forecast_example.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    selection = json.loads((ROOT / "outputs" / "model_selection.json").read_text(encoding="utf-8"))
    for name in ["temporal_tcn","stgnn"]:
        selected_run = selection["selected_by_validation_only"][name]
        hist = pd.read_csv(ROOT / "outputs" / "training" / f"{selected_run}.csv")
        axs[0].plot(hist.epoch,hist.train_scaled_mse,label=name)
        axs[1].plot(hist.epoch,hist.validation_scaled_mae,label=name)
    for ax,title in zip(axs,["Training standardized MSE","Validation standardized MAE"]):
        ax.set(title=title,xlabel="Epoch")
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)
    print("PLOTS_COMPLETE", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all","preprocess","train","plot"], default="all")
    args = parser.parse_args()
    cfg = read_config()
    (ROOT / "outputs").mkdir(exist_ok=True)
    dump_json(ROOT / "outputs" / "run_environment.json", {"python":sys.version, "numpy":np.__version__,
              "pandas":pd.__version__,"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"config":cfg})
    if args.stage in ["all","preprocess"]:
        preprocess(cfg)
    if args.stage in ["all","train"]:
        fit(cfg)
    if args.stage in ["all","plot"]:
        plot_results(cfg)

if __name__ == "__main__":
    main()
