# Ames House Price Prediction

DSAI4204 team project for the [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).



## Notebooks

- `house_prices_v1.ipynb`: Ridge regression with a fixed 80/20 training-validation split. Compares the original features against six additional features.
- `house_prices_v2.ipynb`: Ridge and Gradient Boosting with five-fold cross-validation, plus a fixed 50/50 blend of their log-price predictions. Preprocessing is fitted separately inside each training fold.

Both versions predict natural log prices and use `exp()` to recover the original price scale. Lower log-RMSE is better. The V1 single-split score and V2 cross-validation score use different validation settings and should not be treated as a direct measure of improvement.

## Files

- `Dataset/`: competition training data, test data, sample submission, and field descriptions.
- `requirements-v1.txt`: dependencies shared by both notebooks.
- `submission_v1.csv`: the existing V1 predictions.
- `README-v1.md`: additional V1 instructions.

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
