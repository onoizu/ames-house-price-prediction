# Ames House Price Prediction

DSAI4204 team project for the [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).


## Kaggle notebook

[Open the Kaggle notebook](house_prices_kaggle.ipynb) and attach the Ames training and test CSVs under `Traindata` and `testData`. The notebook reads the raw tables, explores missing facilities, price, size, quality and neighbourhood, and fits SVR, XGBoost and CatBoost. Each validation house receives a prediction from models trained on the other four folds.

The final cells refit the models on all labelled houses, apply the comparable-sale check, and write `/kaggle/working/submission.csv`. This training notebook reports its own five-fold validation error. The project's leaderboard result belongs to the separately archived submission.

## Final integrated notebook

[Open the final notebook](house_prices_final.ipynb) to follow the same five milestones as the project report: V1 starts with Ridge and six features; V2 compares and combines learners; V3 reviews input treatments and model alternatives; V4 introduces neighbourhood context; V5 combines the final predictions and checks comparable sales.

The notebook includes executed EDA plots, the Ridge training run, recorded model comparisons, and the final V5 prediction pipeline. The middle milestones use saved OOF predictions and experiment records. The final combination uses saved component predictions, with intermediate and final checks against the prediction archive; it does not retrain every component.

Run all cells inside a clone containing `Dataset/`, `final_versions/`, and the recorded experiment files. Install `requirements-v1.txt` in the selected Python kernel. The asset manifest, `final_versions/notebook_assets.json`, supplies named inputs and their hashes. Output files are written to `final_outputs/`, including `submission_final.csv`, the baseline CSV, comparison tables, comparable-sale records and a run manifest.

## Notebooks


- `house_prices_v1.ipynb`: Ridge regression with a fixed 80/20 training-validation split. Compares the original features against six additional features.
- `house_prices_v2.ipynb`: Ridge and Gradient Boosting with five-fold cross-validation, plus a fixed 50/50 blend of their log-price predictions. Preprocessing is fitted separately inside each training fold.

Both versions predict natural log prices and use `exp()` to recover the original price scale. Lower log-RMSE is better. The V1 single-split score and V2 cross-validation score use different validation settings and should not be treated as a direct measure of improvement.

The [neighbourhood notebook](house_prices_v4.ipynb) and [its run instructions](README_v4.md) preserve the later neighbourhood recipe. Use the final integrated notebook above for the report's complete five-milestone narrative and final prediction file.

## Files

- `Dataset/`: competition training data, test data, sample submission, and field descriptions.
- `requirements-v1.txt`: dependencies for V1, V2 and the final integrated notebook.

The dataset contains 1,460 labeled training rows and 1,459 test rows. Dataset source: Kaggle House Prices, based on the Ames Housing data compiled by Dean De Cock.

## Run

Use Python 3.10 or newer. From the repository root:

```sh
python3 -m pip install -r requirements-v1.txt
python3 -m jupyterlab
```

Open a notebook, select the appropriate Python kernel, and run its cells in order. V1 and V2 also recognize the official competition data path on Kaggle. No GPU is required.

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
