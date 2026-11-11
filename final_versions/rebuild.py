"""Rebuild the scored V5–V8 submissions from the recorded model predictions."""

import argparse
import csv
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
COMPONENTS = HERE / "components"
SUBMISSIONS = HERE / "submissions"
VERSION_FILES = {
    "V5": "v5.csv",
    "V6": "v6.csv",
    "V7": "v7.csv",
    "V8": "v8.csv",
}


def read_component(name, column, expected_ids):
    frame = pd.read_csv(COMPONENTS / name)
    if not frame["Id"].equals(expected_ids):
        raise ValueError(f"{name} has different house IDs or row order")
    values = frame[column].to_numpy()
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError(f"{name} has invalid predictions")
    return values


def encode_prices(ids, prices):
    return pd.DataFrame({"Id": ids, "SalePrice": prices}).to_csv(index=False).encode("utf-8")


def read_prices(payload):
    return pd.read_csv(BytesIO(payload))["SalePrice"].to_numpy()


def rebuild_v5(ids):
    stack = read_component("v5_stacked_log_predictions.csv", "stacked_log_prediction", ids)
    svm = read_component("v5_svm_predictions.csv", "SalePrice", ids)
    raw = np.expm1(np.sqrt(stack * np.log(svm)))
    low, high = np.quantile(raw, [0.0042, 0.99])
    adjusted = raw.copy()
    adjusted[raw <= low] *= 0.77
    adjusted[raw >= high] *= 1.10
    return encode_prices(ids, adjusted)


def rebuild_v6(ids, v5):
    experiment = read_component("submission_experiment_01.csv", "SalePrice", ids)
    prices = np.exp(0.5 * np.log(read_prices(v5)) + 0.5 * np.log(experiment))
    return encode_prices(ids, prices)


def rebuild_v7(test, v6):
    ids = test["Id"]
    normal = test["SaleCondition"].eq("Normal").to_numpy()
    tabicl = read_component("v7_tabicl_log_predictions.csv", "prediction_log", ids)
    kumo = read_component("v7_kumo_log_predictions.csv", "log_price", ids)
    prices = read_prices(v6).copy()
    intermediate = prices.copy()
    intermediate[normal] = np.exp(0.75 * np.log(prices[normal]) + 0.25 * tabicl[normal])
    # The scored run wrote and reloaded the TabICL blend before adding Kumo.
    intermediate = read_prices(encode_prices(ids, intermediate))
    prices[normal] = np.exp(0.25 * np.log(intermediate[normal]) + 0.75 * kumo[normal])
    return encode_prices(ids, prices)


def large_partial_new_sales(frame):
    age = frame["YrSold"] - frame["YearBuilt"]
    return (
        frame["SaleCondition"].eq("Partial")
        & frame["SaleType"].eq("New")
        & frame["GrLivArea"].gt(4000)
        & age.between(-1, 1)
    )


def rebuild_v8(train, test, v7):
    candidates = train.loc[large_partial_new_sales(train)]
    base = read_prices(v7)
    prices_by_id = {}
    for i in np.flatnonzero(large_partial_new_sales(test).to_numpy()):
        row = test.iloc[i]
        similar = candidates[
            candidates["Neighborhood"].eq(row["Neighborhood"])
            & candidates["BldgType"].eq(row["BldgType"])
            & candidates["OverallQual"].sub(row["OverallQual"]).abs().le(1)
            & np.log(candidates["GrLivArea"] / row["GrLivArea"]).abs().le(0.25)
            & np.log(candidates["LotArea"] / row["LotArea"]).abs().le(0.60)
        ]
        if similar.empty:
            continue
        reference = np.median(np.log(similar["SalePrice"].to_numpy()))
        estimate = np.exp(0.5 * np.log(base[i]) + 0.5 * reference)
        prices_by_id[str(row["Id"])] = f"{estimate:.8f}"

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=["Id", "SalePrice"], lineterminator="\n")
    writer.writeheader()
    for row in csv.DictReader(StringIO(v7.decode("utf-8"))):
        row["SalePrice"] = prices_by_id.get(row["Id"], row["SalePrice"])
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def compare_with_submission(version, payload, ids):
    path = SUBMISSIONS / VERSION_FILES[version]
    original = path.read_bytes()
    frame = pd.read_csv(BytesIO(payload))
    expected = pd.read_csv(BytesIO(original))
    if not frame["Id"].equals(ids) or not expected["Id"].equals(ids):
        raise ValueError(f"{version} has different house IDs or row order")
    np.testing.assert_allclose(
        frame["SalePrice"], expected["SalePrice"], rtol=1e-12, atol=1e-7,
        err_msg=f"{version} differs from the scored submission",
    )
    match = "byte-identical" if payload == original else "numerically matched"
    print(f"{version}: {match}, scored CSV SHA-256 {sha256(original).hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE.parent / "Dataset")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = pd.read_csv(args.data / "train.csv", keep_default_na=False, na_values=["", "NA"])
    test = pd.read_csv(args.data / "test.csv", keep_default_na=False, na_values=["", "NA"])
    if len(train) != 1460 or len(test) != 1459 or not test["Id"].is_unique:
        raise ValueError("Use the competition train.csv and test.csv")

    v5 = rebuild_v5(test["Id"])
    v6 = rebuild_v6(test["Id"], v5)
    v7 = rebuild_v7(test, v6)
    v8 = rebuild_v8(train, test, v7)
    versions = dict(zip(VERSION_FILES, [v5, v6, v7, v8]))
    for version, payload in versions.items():
        compare_with_submission(version, payload, test["Id"])
    args.output.mkdir(parents=True, exist_ok=True)
    for version, payload in versions.items():
        (args.output / VERSION_FILES[version]).write_bytes(payload)


if __name__ == "__main__":
    main()
