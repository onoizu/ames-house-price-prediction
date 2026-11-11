# Final submissions V5–V8

| Version | Run file | Scored submission | Kaggle log RMSE | Main change |
| --- | --- | --- | ---: | --- |
| V5 | [v5.py](v5.py) | [v5.csv](submissions/v5.csv) | 0.11599 | Combine Python stacking with an R MICE/SVM model and adjust the lowest and highest predictions. |
| V6 | [v6.py](v6.py) | [v6.csv](submissions/v6.csv) | 0.11419 | Blend V5 equally with the SVR, CatBoost, and residual correction experiment on the log-price scale. |
| V7 | [v7.py](v7.py) | [v7.csv](submissions/v7.csv) | 0.11194 | Blend V6 with TabICLv2 and two Kumo models for normal sales. Other sale conditions use the V6 predictions. |
| V8 | [v8.py](v8.py) | [v8.csv](submissions/v8.csv) | **0.11101** | Use comparable training sales to adjust the V7 estimate for one exceptionally large new home. |

The [V5 prediction before the tail adjustment](components/submission_candidate_raw.csv) scored 0.11781. The adjustment changes 22 of the 1,459 prices. The [other V6 component](components/submission_experiment_01.csv) scored 0.11646 on its own.

For normal sales, the effective V7 weights are 18.75% V6, 6.25% TabICLv2, 37.5% medium Kumo and 37.5% large Kumo on the log-price scale. The saved Kumo component contains the mean of the two Kumo predictions. V8 matches large homes with `SaleType=New` and `SaleCondition=Partial` to similar labeled sales. Its rule found two training comparables for one test home. The data dictionary defines `Partial` as incomplete when last assessed.

## Rebuild the submissions

From the repository root, run these commands in your Python environment.

```bash
python -m pip install -r final_versions/requirements.txt
python final_versions/v8.py
```

Run `v5.py`, `v6.py`, or `v7.py` for the corresponding CSV. In VS Code, open the version file and choose **Run Python File** after installing the dependencies in the selected Python environment. Run `python final_versions/rebuild.py` to generate all four CSVs. The default output folder is `rebuilt/` at the repository root. Use `--output` to pick a different folder.

The scripts start from fitted-model predictions in [components/](components/) and use the existing `Dataset/` files. They rerun the combination and adjustments and verify the rebuilt files against the scored CSVs. The four CSVs reproduced byte for byte in the recorded NumPy 2.5.4 and pandas 3.1.1 environment.
