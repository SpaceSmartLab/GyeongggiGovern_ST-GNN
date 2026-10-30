"""Forecast known sensors from a tidy CSV using the validation-selected ST-GNN.

Input columns: timestamp, serial, temperature_c, relative_humidity_pct.
Example: python predict.py --input data/processed/clean_observations_all_dates.csv.gz
         --origin "2026-09-21 13:00:00"
Future rows are discarded BEFORE building input features. No retraining occurs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pipeline import ROOT, NODES, VARIABLES, STGNN, causal_fill


def inference_features(frame, origin, checkpoint):
    cfg = checkpoint["config"]
    serials = checkpoint["serials"]
    unknown = set(frame.serial.unique()) - set(serials)
    if unknown:
        raise ValueError(f"Unknown device IDs need reviewed graph changes and validation: {unknown}")
    if origin != origin.floor("10min"):
        raise ValueError("Origin must fall on a 10-minute grid")
    frame = frame.loc[frame.timestamp.le(origin)].copy()
    if frame.duplicated(["timestamp", "serial"]).any():
        raise ValueError("Duplicate timestamp/device keys; use the audited preprocessing pipeline first")
    if not frame.timestamp.eq(frame.timestamp.dt.floor("10min")).all():
        raise ValueError("Off-grid timestamps; use the audited preprocessing pipeline first")
    # Keep enough preceding context to forward-fill the beginning of the input window.
    times = pd.date_range(end=origin, periods=cfg["history_steps"]*2, freq="10min")
    y = np.full((len(times), len(serials), 2), np.nan, np.float32)
    for ni, serial in enumerate(serials):
        rows = frame[frame.serial.eq(serial)].set_index("timestamp")
        y[:, ni] = rows[VARIABLES].reindex(times).to_numpy(np.float32)
    valid_ranges = [(y[...,0] >= -20) & (y[...,0] <= 60), (y[...,1] >= 0) & (y[...,1] <= 100)]
    for v in range(2):
        y[...,v] = np.where(valid_ranges[v], y[...,v], np.nan)
    observed = np.isfinite(y)
    filled, age = np.empty_like(y), np.zeros_like(y)
    for ni in range(len(serials)):
        filled[:, ni] = causal_fill(pd.DataFrame(y[:,ni]), cfg["forward_fill_limit"]).to_numpy()
        for v in range(2):
            elapsed = cfg["history_steps"]
            for ti in range(len(times)):
                elapsed = 0 if observed[ti,ni,v] else elapsed+1
                age[ti,ni,v] = min(elapsed,cfg["history_steps"])/cfg["history_steps"]
    available = np.isfinite(filled)
    mean, std = np.array(checkpoint["mean"], np.float32), np.array(checkpoint["std"], np.float32)
    scaled = np.nan_to_num((filled-mean)/std, nan=0.)
    minute = times.hour.to_numpy()*60+times.minute.to_numpy()
    week = times.dayofweek.to_numpy()+minute/1440.
    calendar = np.stack([np.sin(2*np.pi*minute/1440),np.cos(2*np.pi*minute/1440),
                         np.sin(2*np.pi*week/7),np.cos(2*np.pi*week/7),
                         (times.dayofweek.to_numpy()<5).astype(float)],axis=-1).astype(np.float32)
    features = np.concatenate([scaled,observed.astype(np.float32),available.astype(np.float32),age,
                               np.broadcast_to(calendar[:,None],(len(times),len(serials),5))],axis=-1)
    return features[-cfg["history_steps"]:], available[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="outputs/custom_forecast.csv")
    args = parser.parse_args()
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        selection = json.loads((ROOT/"outputs/model_selection.json").read_text(encoding="utf-8"))
        checkpoint_path = ROOT/"models"/(selection["selected_by_validation_only"]["stgnn"]+".pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    torch.set_num_threads(checkpoint["config"]["torch_threads"])
    frame = pd.read_csv(args.input, dtype={"serial":str})
    missing = set(["timestamp","serial"]+VARIABLES)-set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    frame["timestamp"] = pd.to_datetime(frame.timestamp,format="mixed",errors="raise")
    origin = pd.Timestamp(args.origin)
    features, available = inference_features(frame,origin,checkpoint)
    if not available.all(-1).any():
        raise ValueError("No device has both variables within 20 minutes of the forecast origin")
    graph = checkpoint["state_dict"]["graph"].numpy()
    model = STGNN(checkpoint["features"],graph,checkpoint["config"]["hidden_channels"],len(checkpoint["config"]["horizon_steps"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = torch.from_numpy(features.transpose(2,0,1).copy())[None]
    with torch.no_grad():
        forecast = model(x).numpy()[0]*np.array(checkpoint["std"],np.float32)+np.array(checkpoint["mean"],np.float32)
    rows = []
    labels = {node[0]:node[1] for node in NODES}
    for hi,h in enumerate(checkpoint["config"]["horizon_steps"]):
        for ni,serial in enumerate(checkpoint["serials"]):
            if available[ni].all():
                rows.append({"issued_at":origin,"forecast_time":origin+pd.Timedelta(minutes=h*10),
                             "horizon_minutes":h*10,"serial":serial,"sensor":labels[serial],
                             VARIABLES[0]:forecast[hi,ni,0],VARIABLES[1]:forecast[hi,ni,1]})
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)
    print(f"Saved {len(rows)} forecasts to {path}; checkpoint={checkpoint_path.name}")

if __name__ == "__main__":
    main()
