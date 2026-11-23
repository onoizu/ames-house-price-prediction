# Ames House Price Prediction

Predict residential sale prices in Ames, Iowa, from property characteristics and transaction records. The project follows the modeling decisions from a Ridge baseline through feature engineering, cross-validated model comparisons, and the final blend. The best recorded Kaggle result was **0.11101 log-RMSE**, with predictions for all **1,459** test houses. [Scored submission](final_versions/submissions/v8.csv)

[Final notebook](house_prices_final.ipynb) · [Kaggle run](https://www.kaggle.com/code/anmeihui/ames-house-prices-final-ant12?scriptVersionId=372198732) · [Run locally](#run-locally) · [Repository guide](#repository-guide)

## Data and evaluation

The Kaggle dataset contains **1,460 labeled sales** and **1,459 test houses**. It provides **79 predictors** covering living area, overall quality, building age, neighborhood, facilities, and sale conditions. The target is `SalePrice` in dollars. We use `Id` to align predictions and audit records. [Competition data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) · [Field definitions](Dataset/data_description.txt)

Kaggle evaluates root mean squared error between the natural logs of the observed and predicted prices. Lower values indicate better performance. We train on log prices and convert final predictions back to dollars for submission. [Kaggle evaluation](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation)

We used a fixed 80/20 holdout to compare the two Ridge feature sets, then moved to shuffled five-fold cross-validation during model development. Each out-of-fold (OOF) prediction comes from a model fitted on four of the five folds. Holdout and OOF scores measure performance on different splits. The Kaggle scores below come from recorded submissions.

## Data cleaning and exploratory analysis

- **Read missing values in context.** In the training set, all 1,453 houses with a missing pool rating had zero pool area. In the test set, three houses had a positive pool area and a missing rating. That example shows why related measurements matter when deciding whether a blank indicates an absent facility or an unknown description.
- **Fit preprocessing on training rows.** The Ridge pipeline fills remaining numeric values with medians and categorical values with modes learned from the training split. It uses the training dummy columns and fitted scaler for validation. Later cross-validation runs learn preprocessing statistics within each training fold.
- **Use observations to guide model changes.** Taking logs reduced sale-price skewness from **1.88 to 0.12**. The living-area plot highlighted houses **524** and **1299**, which sold for much less than their size might suggest. Price distributions also varied across neighborhoods, giving us a reason to test features based on local characteristics.

See the distribution plots, missing-value checks, and neighborhood comparisons in the [final notebook](house_prices_final.ipynb).

## Model development

### V1 · Ridge baseline and housing features

After one-hot encoding, the holdout baseline had 326 predictors, including several overlapping measurements of floor space. Ridge's L2 penalty gave us a practical way to handle that overlap. We then added six features while keeping the model and split fixed:

- `TotalSF` sums the basement, first-floor, and second-floor areas. Selected area variables use `log1p` to compress large values while allowing zeros.
- `TotalBath` counts full baths and gives half weight to half baths, including those in the basement.
- `HouseAge` and `RemodelAge` measure time since construction and remodeling at the time of sale.
- `HasGarage` and `HasBasement` indicate whether those facilities are present.

The six features reduced holdout log-RMSE from **0.12428 to 0.12383**. The gain was modest, so we kept the baseline for comparison and tested a learner that could represent nonlinear relationships. [Ridge notebook](house_prices_v1.ipynb) · [Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

### V2 · Ridge and Gradient Boosting

On the same five folds, Ridge scored **0.14004** OOF log-RMSE and Gradient Boosting scored **0.13317**. An equal average of their log-price predictions scored **0.12786**. Ridge had the higher individual error, yet its predictions improved the combined result. That gave us a reason to retain it in the blend. [V2 notebook](house_prices_v2.ipynb) · [Fold results](cv_results_v2.csv)

### V3 · Input treatment and model selection

We refined facility handling, used training-fold neighborhood medians for `LotFrontage`, and added features for finished area, outdoor area, and garage age. To see whether very large measurements were influencing SVR predictions, we kept its hyperparameters fixed and compared three input treatments:

- **No cap:** 0.12634 OOF log-RMSE.
- **99th-percentile cap:** 0.12150.
- **98th-percentile cap:** 0.12244.

The 99th-percentile cap gave the strongest result in that comparison. Cutoffs came from the training fold, and every house stayed in the evaluation. Two follow-up checks helped narrow the choice: robust scaling increased error in the tested SVR configuration, and applying the 99th-percentile cap to XGBoost slightly increased its error. We kept the cap for SVR and used uncapped inputs for XGBoost and CatBoost in the selected V3 blend. [V3 notebook](house_prices_v3.ipynb) · [Candidate results](v3_experiments/all_candidate_results.csv)

### V4 · Neighborhood context

A 2,000-square-foot house can be unusually large in one neighborhood and typical in another. We expressed that difference by dividing each home's living area by the median living area of its neighborhood in the training fold. The model also tested interactions with housing attributes and neighborhood price summaries. For those price summaries, each training row's own sale price was excluded from its encoding, and validation rows used training-fold statistics.

On the two paired five-fold runs, the revised recipe reduced OOF log-RMSE from **0.117905 to 0.114679** and from **0.115709 to 0.113709**. It improved **9 of 10** individual fold comparisons. These results compare the full recipes, so they support keeping the neighborhood model as a whole. [V4 notebook](house_prices_v4.ipynb) · [V4 training notes](README_v4.md)

### V5 · Final blend and comparable sales

The final pipeline rebuilt the earlier stacking and SVM combination, then blended it with predictions from the SVR/CatBoost pipeline. For the **1,204** rows marked `SaleCondition=Normal`, we combined the preceding blend with saved TabICLv2 and Kumo predictions on the log-price scale. The effective weights were **18.75%** for the preceding blend, **6.25%** for TabICLv2, and **75%** for the averaged Kumo component. The other **255** rows started from the preceding blend. [Submission archive](final_versions/README.md)

We then checked a narrowly defined group of large new homes marked `Partial`, which means incomplete when last assessed. Test house **2550** matched two labeled sales (**524** and **1299**) based on neighborhood, building type, quality, and size. Taking an equal average of the model's log estimate and the median log price of those comparable sales moved the estimate from **$339,331 to $241,539**. The other **1,458** predictions stayed the same. The rule matched one test house, so its effect is specific to this dataset. [Final notebook](house_prices_final.ipynb) · [Rebuild code](final_versions/rebuild.py)

## Recorded Kaggle results

- **V3 model:** 0.11881 log-RMSE.
- **V4 neighborhood model:** 0.11735.
- **Final refinements:** the score moved from **0.11419** to **0.11194** after changing the blend for normal sales. The comparable-sale adjustment brought it to **0.11101**.

These scores are the recorded results for the competition dataset. The local OOF results informed model and feature selection, and the second V4 split checked stability under a different partition of the same labeled data. [V4 results](README_v4.md) · [Final submission results](final_versions/README.md)

## Run locally

### Explore the workflow in Jupyter

Use Python 3.13 to match the environment used to verify the integrated notebook. From a terminal:

```bash
git clone https://github.com/onoizu/ames-house-price-prediction.git
cd ames-house-price-prediction
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-v1.txt
python -m jupyterlab
```

On Windows PowerShell, substitute `py -3.13 -m venv .venv` and `.\.venv\Scripts\Activate.ps1` for the environment creation and activation commands.

Open `house_prices_final.ipynb` from the repository root and choose **Run All Cells** using the new environment's Python kernel. The repository provides the data and prediction inputs, so the notebook needs neither a Kaggle API key nor a GPU. It fits the Ridge baseline, reads saved OOF results for the development comparisons, and rebuilds the final submission from archived component predictions. The individual training notebooks have their own dependency files and run instructions.

### Rebuild the scored predictions without Jupyter

From the repository root, using an activated Python environment:

```bash
python -m pip install -r final_versions/requirements.txt
python final_versions/v8.py
```

This writes `rebuilt/v8.csv` and compares it with the archived scored CSV. To rebuild every archived checkpoint, run `python final_versions/rebuild.py`. The archive scripts reuse saved model predictions. The final blend can be reproduced through this path without fitting its component models. [Archive instructions](final_versions/README.md)

## Notebook outputs

The integrated notebook writes to `final_outputs/`:

- `submission_final.csv` and `submission_ridge_baseline.csv` contain predictions with the Kaggle columns `Id` and `SalePrice`.
- `final_prediction_checks.csv` records agreement with the intermediate and final archived predictions. Checks cover test row order and numerical agreement within floating-point tolerance. The final export also checks for positive finite predictions.
- `ridge_feature_comparison.csv`, `model_review_v2.csv`, `input_cap_review.csv`, and `neighbourhood_review.csv` capture the results behind the main modeling decisions.
- `comparable_adjustments.csv` and `comparable_training_sales.csv` show which estimate changed and the labeled sales used to adjust it.
- `run_manifest_final.json` records software versions, data and input hashes, and the verification status.

## Repository guide

- [**Integrated notebook**](house_prices_final.ipynb) — Start here for the EDA, the modeling decisions, and the final CSV.
- [**Dataset**](Dataset/) — Training data, test data, [sample submission](Dataset/sample_submission.csv), and the data dictionary.
- [**V1**](house_prices_v1.ipynb) and [**V2**](house_prices_v2.ipynb) — Ridge feature comparison and five-fold Ridge/Gradient Boosting blend. Use [requirements-v1.txt](requirements-v1.txt).
- [**V3**](house_prices_v3.ipynb) and [**experiment records**](v3_experiments/) — SVR, boosted trees, and the input-treatment comparisons. Use [requirements-v3.txt](requirements-v3.txt).
- [**V4**](house_prices_v4.ipynb) — Neighborhood-aware model and paired validation. See the [training instructions](README_v4.md) and [dependencies](requirements_v4.txt).
- [**Final submission archive**](final_versions/) — Saved component predictions, scored submissions, rebuild scripts, and the [asset manifest](final_versions/notebook_assets.json) used by the integrated notebook.

## References

- Dean De Cock, 2011. [Original Ames Housing dataset paper](https://www.tandfonline.com/doi/abs/10.1080/10691898.2011.11889627).
- [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
- [scikit-learn documentation on cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
