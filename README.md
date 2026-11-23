# Ames House Price Prediction

A machine-learning workflow for predicting residential sale prices in Ames, Iowa, using the [Kaggle House Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). It covers data cleaning, exploratory analysis, feature engineering, model comparison, and reconstruction of the final submission.

**Best recorded Kaggle score:** `0.11101` log-RMSE ([scored predictions](final_versions/submissions/v8.csv)).

[Start with the final notebook](house_prices_final.ipynb) · [View the executed Kaggle notebook](https://www.kaggle.com/code/anmeihui/ames-house-prices-final-ant12?scriptVersionId=372198732)

## Data and evaluation

The dataset contains **1,460 labeled sales**, **1,459 test houses**, and **79 predictors** spanning location, area, quality, age, facilities, and sale condition. `SalePrice` is the target, and `Id` links predictions back to the test rows. The [data dictionary](Dataset/data_description.txt) describes the fields and their categories.

Kaggle uses root mean squared error on the natural logarithms of the predicted and observed prices. Lower is better. The notebook identifies the validation setup for each comparison and reports leaderboard scores separately from local validation results.

## Modeling workflow

1. **Inspect and prepare the data.** We compared missing facility ratings with related fields to distinguish absent facilities from unknown ratings. For validation, we estimated imputation and scaling statistics from the training split or fold and applied them to its held-out rows. The price distribution and the relationship between living area and sale price guided the log transformation and later checks on unusually large homes.
2. **Build a Ridge baseline.** We compared the original predictors with six engineered features under the same holdout split. One of those features, `TotalSF`, combines basement, first-floor, and second-floor areas.
3. **Compare models and limit extreme inputs.** We used out-of-fold predictions to compare Ridge, Gradient Boosting, and their blend. Later experiments evaluated SVR, XGBoost, and CatBoost. For SVR, capping large continuous values at training-fold quantiles reduced their influence while keeping every row. The 99th-percentile cap performed better than the 98th-percentile cap in the recorded comparison.
4. **Add neighborhood context.** A house's living area relative to the neighborhood median helps represent how large it is for its location. The overall neighborhood model improved validation error on both five-fold splits and in nine of the ten paired fold comparisons.
5. **Combine the final predictions.** We used different blending rules by sale condition and applied a comparable-sale adjustment to one test house. The final notebook verifies the reconstructed predictions against archived submission checkpoints.

## Run locally

The integrated notebook was tested with Python 3.13. Run these commands from a terminal:

```bash
git clone https://github.com/onoizu/ames-house-price-prediction.git
cd ames-house-price-prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-v1.txt
python -m jupyterlab
```

Open `house_prices_final.ipynb` from the repository root and run all cells. The clone includes the input data and prediction artifacts needed for this notebook. The run fits the Ridge baseline, reviews saved development results, and rebuilds the final submission from archived component predictions. Full retraining of the final ensemble is outside the notebook's scope.

## Outputs

The integrated notebook writes these files under `final_outputs/`:

- `submission_final.csv` — final predictions with columns `Id` and `SalePrice`.
- `submission_ridge_baseline.csv` — predictions from the Ridge baseline.
- `final_prediction_checks.csv` — comparisons with archived submission checkpoints.
- `run_manifest_final.json` — package versions, input hashes, and verification results.
- Model comparison tables and records for the comparable-sale adjustment.

## Repository guide

- [Integrated notebook](house_prices_final.ipynb) — EDA, Ridge baseline, and final prediction workflow.
- [Ridge baseline](house_prices_v1.ipynb) and [Ridge–Gradient Boosting blend](house_prices_v2.ipynb) — early model development.
- [SVR and boosted-tree blend](house_prices_v3.ipynb) and [neighborhood model](house_prices_v4.ipynb) — later training notebooks. See the [V3 dependencies](requirements-v3.txt) and [V4 run instructions](README_v4.md).
- [Experiment records](v3_experiments/) — candidate comparisons and validation checks.
- [Submission archive](final_versions/) — saved model predictions, scored CSVs, and scripts for reconstructing submissions. The [asset manifest](final_versions/notebook_assets.json) records the files and checksums used by the integrated notebook.

## References

- [Kaggle House Prices evaluation](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation)
- [scikit-learn Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
