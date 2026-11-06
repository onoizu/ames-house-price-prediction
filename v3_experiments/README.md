# V3 experiment record

The final notebook is `../house_prices_v3.ipynb`; the Kaggle submission is `../submission_v3.csv`.

- `all_candidate_results.csv`: results for all 69 model/preprocessing candidates on the same five seed-42 folds.
- `all_model_specs.json`: the exact candidate parameters.
- `selected_config.json`: locked three-model blend and weights selected on a coarse 0.1 grid.
- `selection_oof.csv`: V2 and V3 out-of-fold predictions on the selection folds.
- `stability_check.csv` and `audit_oof.csv`: results on a second set of five folds, seed 2026, without changing the locked blend.
- `model_search.csv`, `cat_search.csv`, and `fold_scores.csv`: detailed search records. Search time includes preprocessing overhead; use it as an approximate computational-cost record.
- `*.log`: actual execution logs.
- `run_search.py`: rerun the candidate comparison from the project files. Run from the repository root using the V3 dependencies. It overwrites search logs/tables but does not retune the locked final blend automatically.

Preprocessing is fitted within each training fold. All 1,460 validation predictions are retained, including the two unusually large, low-priced houses. Continuous feature clipping changes feature values, not target prices, and does not remove observations. No hidden test labels or external training data are used.

The selection score is not an unbiased estimate after searching candidates and blend weights. The second seed checks stability but reuses the same dataset; it is not an independent final test. Kaggle results must be recorded separately after an actual submission.
