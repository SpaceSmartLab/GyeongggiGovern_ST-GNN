"""Post-run checks independent of training, including source immutability and MAE."""
import hashlib
import json
import unittest
import numpy as np
import pandas as pd
from pipeline import ROOT, read_config, VARIABLES, dump_json

def main():
    cfg = read_config()
    inv = pd.read_csv(ROOT/"outputs/audit/file_inventory.csv")
    source = ROOT.__class__(cfg["source_path"])
    checks = {}
    checks["all_23_source_csv_hashes_unchanged"] = all(hashlib.sha256((source/row.source_file).read_bytes()).hexdigest()==row.sha256 for row in inv.itertuples())
    metrics = pd.read_csv(ROOT/"outputs/metrics.csv")
    m = metrics.query("split == 'test' and period == 'all' and sensor == 'ALL'")
    with np.load(ROOT/"outputs/test_predictions.npz") as z:
        ok = True
        for model in ["persistence","stgnn","temporal_tcn","ridge_spatial"]:
            for hi,h in enumerate(cfg["horizon_steps"]):
                for vi,var in enumerate(VARIABLES):
                    valid = z["mask"][:,hi,:,vi]
                    err = z[model][:,hi,:,vi][valid]-z["target"][:,hi,:,vi][valid]
                    row = m[(m.model==model)&(m.horizon_minutes==10*h)&(m.variable==var)].iloc[0]
                    ok &= np.isclose(np.abs(err).mean(),row.mae,atol=1e-7)
                    ok &= len(err)==row.n
        checks["independent_mae_and_sample_counts_match"] = bool(ok)
    selection = json.loads((ROOT/"outputs/model_selection.json").read_text(encoding="utf-8"))
    for model in ["stgnn","temporal_tcn"]:
        best = min([r for r in selection["runs"] if r["name"]==model],key=lambda x:x["validation_scaled_mae"])
        checks[model+"_selection_matches_validation_minimum"] = selection["selected_by_validation_only"][model]==f"{model}_seed{best['seed']}"
    forecast = pd.read_csv(ROOT/"outputs/cutoff_forecast.csv")
    checks["cutoff_forecast_has_13_sensors_x_3_horizons"] = len(forecast)==39 and forecast.sensor.nunique()==13
    checks["cutoff_forecasts_finite"] = bool(np.isfinite(forecast[VARIABLES].to_numpy()).all())
    checks["all_expected_figures_exist"] = all((ROOT/"outputs/figures"/(name+".png")).is_file() for name in ["coverage","graph","test_mae","learning_curves","forecast_example"])
    suite = unittest.defaultTestLoader.loadTestsFromName("test_pipeline")
    with (ROOT/"outputs/test_results.txt").open("w",encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    checks["unit_and_integration_tests_passed"] = result.wasSuccessful()
    dump_json(ROOT/"outputs/verification.json",{"checks":checks,"tests_run":result.testsRun,
              "failures":len(result.failures),"errors":len(result.errors),"all_passed":all(checks.values())})
    print(json.dumps(checks,indent=2))
    if not all(checks.values()):
        raise SystemExit("Verification failed")

if __name__ == "__main__":
    main()
