# Ames House Price Prediction

DSAI4204 team project for the [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).


## Kaggle result

The final V8 submission scored   **0.11101** log RMSE. See the [V5–V8 submissions and rebuild instructions](final_versions/README.md) for the scored files and their model combinations.

## Notebooks


- `house_prices_v1.ipynb`: Ridge regression with a fixed 80/20 training-validation split. Compares the original features against six additional features.
- `house_prices_v2.ipynb`: Ridge and Gradient Boosting with five-fold cross-validation, plus a fixed 50/50 blend of their log-price predictions. Preprocessing is fitted separately inside each training fold.

Both versions predict natural log prices and use `exp()` to recover the original price scale. Lower log-RMSE is better. The V1 single-split score and V2 cross-validation score use different validation settings and should not be treated as a direct measure of improvement.

The later versions can be opened in Jupyter or VS Code. These notebooks rebuild the scored submissions from saved component predictions and the repository data.

- [V5 notebook](house_prices_v5.ipynb) — stacked predictions and tail adjustment
- [V6 notebook](house_prices_v6.ipynb) — equal blend of two prediction pipelines
- [V7 notebook](house_prices_v7.ipynb) — sale-condition blend
- [V8 notebook](house_prices_v8.ipynb) — comparable-sale adjustment

## Files

- `Dataset/`: competition training data, test data, sample submission, and field descriptions.
- `requirements-v1.txt`: dependencies shared by both notebooks.

The dataset contains 1,460 labeled training rows and 1,459 test rows. Dataset source: Kaggle House Prices, based on the Ames Housing data compiled by Dean De Cock.

## Run

Use Python 3.10 or newer. From the repository root:

```sh
python3 -m pip install -r requirements-v1.txt
python3 -m jupyterlab
```

Open a notebook, select the appropriate Python kernel, and run its cells in order. The notebooks also recognize the official competition data path on Kaggle. No GPU is required.

Running V2 saves `submission_v2.csv`, `cv_results_v2.csv`, `model_comparison_v2.csv`, and `oof_predictions_v2.csv` in the working directory. Review the validation results before uploading predictions to Kaggle.

## Experiment principles

Keep the validation setting consistent when comparing alternatives. Record unsuccessful experiments as well as improvements. Adding features can increase redundancy, computation, and overfitting risk, so retain changes based on validation evidence.

## References

- [Competition and evaluation](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/description)
- `Dataset/data_description.txt`

## V3: selected prediction model

`house_prices_v3.ipynb` contains an executed three-model blend selected from the recorded experiments: 50% SVR, 10% XGBoost, and 40% CatBoost. It improves missing-value handling and quality/area/age features, and limits extreme feature leverage for SVR using training-fold quantiles. All training and validation rows are retained.

Use Python 3.11-3.13 and install the additional dependencies with `python3 -m pip install -r requirements-v3.txt`. Run the notebook to reproduce `submission_v3.csv`. Search parameters, failed trials, and the second-seed stability check are preserved in `v3_experiments/`.

OOF selection scores reuse the data for model selection and must not be described as Kaggle scores or an independent final test.
