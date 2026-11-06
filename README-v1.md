# House Prices - Version 1

Open `house_prices_v1.ipynb` and run the cells from top to bottom.
All explanations and comments are in English. The notebook uses plain pandas and scikit-learn code.

## Run locally

From the `Proj` directory:

```sh
python3 -m pip install -r requirements-v1.txt
python3 -m jupyterlab house_prices_v1.ipynb
```

Keep `train.csv` and `test.csv` in `Dataset/`. On Kaggle, add the official House Prices competition data. No GPU is required.

## Workflow

Explore the data, clean missing values, add six features, encode categories, and fit Ridge regression with alpha=10.
Use the same 80/20 validation split to compare Ridge with and without the six new features. Keep the feature set with the lower validation log-RMSE.
Then refit Ridge on all 1,460 labeled training rows and predict all 1,459 test rows.
Medians, modes, dummy categories, and scalers are learned from training rows only.

## Files produced when you run the notebook

- `submission_v1.csv`: predictions in the original price scale.
- `feature_comparison_v1.csv`: Ridge results with and without the added features.

Restart the kernel and run all cells after this update to refresh the Ridge-only results and submission. Uploading to Kaggle is a separate manual step.

Flow reference: [WWWITTT - blend model](https://www.kaggle.com/code/wwwittt/blend-model).
