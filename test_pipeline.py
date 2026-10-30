"""Checks for leakage, missing-target handling, source precedence and inference."""
import unittest
import numpy as np
import pandas as pd
import torch
from pipeline import (ROOT, NODES, STGNN, causal_fill, deduplicate, load_data, make_graph,
                      read_config, split_origins, target_arrays, Windows, predict_network)

class UnitChecks(unittest.TestCase):
    def test_fill_is_causal_and_bounded(self):
        frame = pd.DataFrame({"x": [1., np.nan, np.nan, np.nan, 9., np.nan]})
        actual = causal_fill(frame, 2).x.to_numpy()
        np.testing.assert_allclose(actual, [1., 1., 1., np.nan, 9., 9.], equal_nan=True)
        changed = frame.copy()
        changed.loc[4, "x"] = 999.
        np.testing.assert_allclose(causal_fill(changed, 2).iloc[:4], causal_fill(frame, 2).iloc[:4], equal_nan=True)

    def test_deduplication_prefers_v2_but_retains_v1_only_records(self):
        data = pd.DataFrame({"serial": ["a"]*3,
                             "timestamp": pd.to_datetime(["2026-08-01", "2026-08-01", "2026-07-31"]),
                             "collection": [1, 2, 1], "source_file": ["first", "second", "first"],
                             "source_row": [2, 2, 3], "value": [1., 1.001, 5.]})
        result = deduplicate(data)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[-1].value, 1.001)
        self.assertEqual(result.iloc[0].value, 5.)

    def test_unknown_location_has_no_invented_neighbors(self):
        graph = make_graph()
        i = [n[1] for n in NODES].index("S18")
        self.assertEqual(np.count_nonzero(graph[i]), 1)
        self.assertEqual(graph[i, i], 1)
        np.testing.assert_allclose(graph, graph.T)

    def test_inactive_neighbor_does_not_affect_forecast(self):
        torch.manual_seed(3)
        model = STGNN(13, np.ones((2, 2)), hidden=4)
        torch.nn.init.normal_(model.head.weight)
        x = torch.randn(1, 13, 18, 2)
        x[:, 4:6, :, 0] = 1
        x[:, 4:6, :, 1] = 0
        changed = x.clone()
        changed[:, :4, :, 1] += 10000
        with torch.no_grad():
            np.testing.assert_allclose(model(x).numpy()[:, :, 0], model(changed).numpy()[:, :, 0], atol=1e-6)

class DataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.data = read_config(), load_data()
        cls.splits = split_origins(cls.data, cls.cfg)

    def test_scaler_uses_only_training_observations(self):
        train = self.data["times"] < np.datetime64(self.cfg["validation_start"])
        np.testing.assert_allclose(self.data["mean"], np.nanmean(self.data["y"][train], axis=(0, 1)))
        np.testing.assert_allclose(self.data["std"], np.nanstd(self.data["y"][train], axis=(0, 1)))

    def test_target_times_do_not_cross_split_boundaries(self):
        ts = self.data["times"]
        h = np.array(self.cfg["horizon_steps"])
        sets = {}
        for name, origins in self.splits.items():
            self.assertGreater(len(origins), 0)
            sets[name] = set(ts[origins[:, None]+h].ravel())
        self.assertTrue(sets["train"].isdisjoint(sets["validation"]))
        self.assertTrue(sets["validation"].isdisjoint(sets["test"]))
        self.assertLess(max(sets["train"]), np.datetime64(self.cfg["validation_start"]))
        self.assertLess(max(sets["validation"]), np.datetime64(self.cfg["test_start"]))

    def test_no_s07_test_labels_and_no_imputed_targets(self):
        target, mask = target_arrays(self.data, self.splits["test"], self.cfg)
        ni = list(self.data["labels"]).index("S07")
        self.assertFalse(mask[:, :, ni].any())
        self.assertTrue(np.isfinite(target[mask]).all())
        expected = np.isfinite(self.data["y"])
        np.testing.assert_array_equal(expected, self.data["observed"])

    def test_model_shape_and_finite_output(self):
        x, y, mask = Windows(self.data, self.splits["train"], self.cfg)[100]
        model = STGNN(13, self.data["graph"], hidden=4)
        out = model(x[None])
        self.assertEqual(tuple(out.shape), (1, 3, len(NODES), 2))
        self.assertTrue(torch.isfinite(out).all())

    def test_standalone_inference_matches_training_features_and_ignores_future(self):
        from predict import inference_features
        frame = pd.read_csv('data/processed/clean_observations_all_dates.csv.gz', dtype={"serial":str})
        frame["timestamp"] = pd.to_datetime(frame.timestamp, format="mixed")
        origin = pd.Timestamp("2026-09-15 12:00:00")
        checkpoint = {"config":self.cfg,"serials":self.data["serials"].tolist(),
                      "mean":self.data["mean"].tolist(),"std":self.data["std"].tolist()}
        features, _ = inference_features(frame,origin,checkpoint)
        i = np.where(self.data["times"] == origin.to_datetime64())[0][0]
        np.testing.assert_allclose(features,self.data["x"][i-17:i+1],atol=1e-7)
        frame.loc[frame.timestamp.gt(origin), "temperature_c"] = -999
        frame.loc[frame.timestamp.gt(origin), "relative_humidity_pct"] = 999
        changed, _ = inference_features(frame,origin,checkpoint)
        np.testing.assert_array_equal(features,changed)

    def test_final_timestamp_prediction_does_not_require_future_labels(self):
        model = STGNN(13,self.data["graph"],hidden=4)
        pred = predict_network(model,self.data,np.array([len(self.data["times"])-1]),self.cfg)
        self.assertEqual(pred.shape,(1,3,len(NODES),2))
        self.assertTrue(np.isfinite(pred).all())

    @unittest.skipUnless((ROOT/"outputs/test_predictions.npz").is_file(), "Run after training")
    def test_saved_checkpoint_reproduces_held_out_predictions(self):
        import json
        selected = json.loads((ROOT/"outputs/model_selection.json").read_text(encoding="utf-8"))["selected_by_validation_only"]["stgnn"]
        ckpt = torch.load(ROOT/"models"/(selected+".pt"),map_location="cpu",weights_only=True)
        model = STGNN(ckpt["features"],ckpt["state_dict"]["graph"].numpy(),ckpt["config"]["hidden_channels"])
        model.load_state_dict(ckpt["state_dict"])
        with np.load(ROOT/"outputs/test_predictions.npz") as predictions:
            indices = np.array([0,100,len(predictions["origins"])-1])
            actual = predict_network(model,self.data,predictions["origins"][indices],self.cfg)
            expected = predictions["stgnn"][indices]
            np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-6)

if __name__ == "__main__":
    unittest.main(verbosity=2)
