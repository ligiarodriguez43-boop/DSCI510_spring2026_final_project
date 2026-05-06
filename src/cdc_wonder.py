"""
CDC WONDER API: California colon cancer mortality (2018-2022).

Fetches age/sex/race/ethnicity-stratified death counts and trains regression
models to predict deaths from demographic features.
"""
import re
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from config import CDC_WONDER_URL, RANDOM_STATE, HTTP_TIMEOUT, TEST_SIZE


# California, colorectal cancer (codes 21041-21052), 2018-2022, grouped by Year x Age x Sex x Race x Ethnicity
QUERY_XML = """<?xml version="1.0" encoding="UTF-8"?><request-parameters>
    <parameter><name>accept_datause_restrictions</name><value>true</value></parameter>
    <parameter><name>B_1</name><value>D207.V1</value></parameter>
    <parameter><name>B_2</name><value>D207.V5</value></parameter>
    <parameter><name>B_3</name><value>D207.V9</value></parameter>
    <parameter><name>B_4</name><value>D207.V4</value></parameter>
    <parameter><name>B_5</name><value>D207.V6</value></parameter>
    <parameter><name>F_D207.V11</name><value>*All*</value></parameter>
    <parameter><name>I_D207.V11</name><value>*All* (The United States)
</value></parameter>
    <parameter><name>M_1</name><value>D207.M1</value></parameter>
    <parameter><name>M_2</name><value>D207.M2</value></parameter>
    <parameter><name>O_PR</name><value>false</value></parameter>
    <parameter><name>O_V11_fmode</name><value>freg</value></parameter>
    <parameter><name>O_cancer</name><value>D207.V8</value></parameter>
    <parameter><name>O_export-format</name><value>tsv</value></parameter>
    <parameter><name>O_javascript</name><value>on</value></parameter>
    <parameter><name>O_location</name><value>D207.V2</value></parameter>
    <parameter><name>O_precision</name><value>1</value></parameter>
    <parameter><name>O_rate_per</name><value>100000</value></parameter>
    <parameter><name>O_show_totals</name><value>false</value></parameter>
    <parameter><name>O_stdpop</name><value>201</value></parameter>
    <parameter><name>O_timeout</name><value>600</value></parameter>
    <parameter><name>O_title</name><value/></parameter>
    <parameter><name>V_D207.V1</name>
        <value>2018</value><value>2019</value><value>2020</value>
        <value>2021</value><value>2022</value></parameter>
    <parameter><name>V_D207.V10</name><value>21041-21052</value></parameter>
    <parameter><name>V_D207.V11</name><value/></parameter>
    <parameter><name>V_D207.V2</name><value>06</value></parameter>
    <parameter><name>V_D207.V3</name><value>*All*</value></parameter>
    <parameter><name>V_D207.V4</name><value>*All*</value></parameter>
    <parameter><name>V_D207.V5</name><value>*All*</value></parameter>
    <parameter><name>V_D207.V6</name><value>*All*</value></parameter>
    <parameter><name>V_D207.V8</name><value>21041-21052</value></parameter>
    <parameter><name>V_D207.V9</name><value>*All*</value></parameter>
    <parameter><name>action-Send</name><value>Send</value></parameter>
    <parameter><name>dataset_code</name><value>D207</value></parameter>
    <parameter><name>dataset_id</name><value>D207</value></parameter>
    <parameter><name>dataset_label</name><value>United States Cancer Statistics, 2018-2023 Mortality Single race</value></parameter>
    <parameter><name>dataset_vintage_latest</name><value>Cancer Mortality Single Race</value></parameter>
    <parameter><name>finder-stage-D207.V11</name><value>codeset</value></parameter>
    <parameter><name>saved_id</name><value/></parameter>
    <parameter><name>stage</name><value>request</value></parameter>
</request-parameters>"""


def fetch_wonder_response(url=CDC_WONDER_URL):
    resp = requests.post(
        url,
        data={"request_xml": QUERY_XML, "accept_datause_restrictions": "true"},
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:1500]}")
    return resp.text

# Parses WONDER's table, handling row-spans, into a DataFrame
def parse_wonder_response(html_text: str) -> pd.DataFrame:
    soup = BeautifulSoup(html_text, "html.parser")
    table = soup.find("data-table")
    if table is None:
        raise RuntimeError("No <data-table> element in response.")

    rows = []
    carried = []
    for r in table.find_all("r"):
        row = [lbl for lbl, _ in carried]
        carried = [[lbl, n - 1] for lbl, n in carried if n - 1 > 0]

        for c in r.find_all("c"):
            label = c.get("l")
            value = c.get("v")
            cell_text = label if label is not None else (value or "")
            row.append(cell_text)

            rspan = c.get("r")
            if rspan and label is not None:
                try:
                    n = int(rspan)
                    if n > 1:
                        carried.append([label, n - 1])
                except ValueError:
                    pass
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.shape[1] != 6:
        raise RuntimeError(f"Expected 6 columns, got {df.shape[1]}")
    df.columns = ["Year", "Age", "Sex", "Race", "Ethnicity", "Deaths"]

    df = df[df["Year"].astype(str).str.match(r"^\d{4}$", na=False)].copy()
    df["Year"] = df["Year"].astype(int)
    df["Deaths"] = df["Deaths"].astype(str).str.replace(",", "", regex=False)
    df["Suppressed"] = (
        df["Deaths"].str.contains(r"[A-Za-z]", regex=True, na=False)
        | (df["Deaths"] == "")
    )
    df.loc[df["Suppressed"], "Deaths"] = None
    df["Deaths"] = pd.to_numeric(df["Deaths"], errors="coerce")
    for col in ["Age", "Sex", "Race", "Ethnicity"]:
        df[col] = df[col].astype(str).str.strip()
    return df

# fetch + parse in one call
def fetch_wonder_data() -> pd.DataFrame:
    return parse_wonder_response(fetch_wonder_response())

# Converts an age range like '65-69 years' to a numeric midpoint
def age_to_midpoint(age_str: str) -> float:
    age_str = age_str.strip()
    if "85+" in age_str:
        return 87.0
    m = re.match(r"(\d+)-(\d+)", age_str)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2
    return np.nan

# One-hot encode and produce X, y for modeling
def build_features(df_usable: pd.DataFrame):
    X = pd.DataFrame()
    X["Year"]         = df_usable["Year"].astype(int)
    X["Age_midpoint"] = df_usable["Age"].apply(age_to_midpoint)
    X["Sex_Male"]     = (df_usable["Sex"] == "Male").astype(int)
    X["Hispanic"]     = (df_usable["Ethnicity"] == "Hispanic").astype(int)
    race_dummies = pd.get_dummies(df_usable["Race"],
                                  prefix="Race", drop_first=True).astype(int)
    X = pd.concat([X, race_dummies], axis=1).reset_index(drop=True)
    y = df_usable["Deaths"].astype(float).reset_index(drop=True)
    return X, y


# EDA plots
def plot_yearly_totals(df_usable):
    plt.figure(figsize=(8, 5))
    yearly = df_usable.groupby("Year")["Deaths"].sum()
    plt.plot(yearly.index, yearly.values, marker="o", linewidth=2,
             color="steelblue", markersize=8)
    plt.title("Colon Cancer Deaths in California (2018-2022)")
    plt.xlabel("Year")
    plt.ylabel("Total Deaths")
    plt.grid(alpha=0.3)
    plt.xticks(yearly.index)
    plt.tight_layout()
    plt.show()


def plot_age_boxplot(df_usable):
    df_usable.boxplot(column="Deaths", by="Age", figsize=(10, 5))
    plt.title("Colon Cancer Deaths by Age Groups in CA (2018-2022)")
    plt.suptitle("")
    plt.xlabel("Age group")
    plt.ylabel("Deaths")
    plt.xticks(rotation=45, fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_sex_race(df_usable):
    pivot = df_usable.groupby(["Sex", "Race"])["Deaths"].sum().unstack()
    pivot.plot(kind="bar", figsize=(8, 5),
               color=["crimson", "steelblue", "mediumseagreen"])
    plt.title("Colon Cancer Deaths by Sex & Race in CA (2018-2022)")
    plt.ylabel("Total deaths")
    plt.xlabel("Sex")
    plt.xticks(rotation=0)
    plt.legend(title="Race", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_ethnicity_trend(df_usable):
    trend = df_usable.groupby(["Year", "Ethnicity"])["Deaths"].sum().unstack()
    trend.plot(figsize=(8, 5), marker="o",
               color=["orange", "indigo"], linewidth=2)
    plt.title("Colon Cancer Death Trends by Ethnicity in CA (2018-2022)")
    plt.xlabel("Year")
    plt.ylabel("Total deaths")
    plt.grid(alpha=0.3)
    plt.legend(title="Ethnicity")
    plt.tight_layout()
    plt.show()


# Machine Learning Modeling - train Linear, RF, GB regressors with 5-fold CV & Returns list of dicts
def train_models(X, y):
    from config import CV_SPLITS

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    X_full_scaled  = StandardScaler().fit_transform(X)

    specs = [
        ("Linear Regression",
         LinearRegression(),
         X_train_scaled, X_test_scaled, X_full_scaled),
        ("Random Forest",
         RandomForestRegressor(n_estimators=200,
                               random_state=RANDOM_STATE, n_jobs=-1),
         X_train, X_test, X),
        ("Gradient Boosting",
         GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                   max_depth=4, random_state=RANDOM_STATE),
         X_train, X_test, X),
    ]

    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for name, model, Xtr, Xte, Xfull in specs:
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        cv_r2 = cross_val_score(model, Xfull, y, cv=cv, scoring="r2")
        cv_mae = -cross_val_score(model, Xfull, y, cv=cv,
                                  scoring="neg_mean_absolute_error")
        results.append({
            "Model":       name,
            "Test_R2":     r2_score(y_test, preds),
            "Test_MAE":    mean_absolute_error(y_test, preds),
            "Test_RMSE":   np.sqrt(mean_squared_error(y_test, preds)),
            "CV_R2_mean":  cv_r2.mean(),
            "CV_R2_std":   cv_r2.std(),
            "CV_MAE_mean": cv_mae.mean(),
            "CV_MAE_std":  cv_mae.std(),
            "predictions": preds,
            "fitted_model": model,
        })

    return results, (X_train, X_test, y_train, y_test, scaler)


def plot_predicted_vs_actual(results, y_test):
    for res in results:
        plt.figure(figsize=(7, 6))
        plt.scatter(y_test, res["predictions"], alpha=0.6,
                    color="steelblue", s=40)
        lim = max(y_test.max(), max(res["predictions"])) * 1.05
        plt.plot([0, lim], [0, lim], color="red",
                 linestyle="--", linewidth=1.5, label="Perfect prediction")
        plt.xlabel("Actual deaths")
        plt.ylabel("Predicted deaths")
        plt.title(f"{res['Model']}\n"
                  f"R^2 = {res['Test_R2']:.3f}, "
                  f"MAE = {res['Test_MAE']:.1f} deaths")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_feature_importance(results, X):
    colors = {"Random Forest": "mediumseagreen",
              "Gradient Boosting": "orange"}
    for name in ["Random Forest", "Gradient Boosting"]:
        model = next(r for r in results
                     if r["Model"] == name)["fitted_model"]
        imp = pd.Series(model.feature_importances_,
                        index=list(X.columns)).sort_values()
        imp.plot(kind="barh", figsize=(8, 5), color=colors[name])
        plt.title(f"{name} - Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

# Predict deaths for a demographic-group/year profile
def predict_new_patient(results, X, scaler,
                        new_patient=None):
    if new_patient is None:
        new_patient = {
            "Year": 2022,
            "Age": "65-69 years",
            "Sex": "Male",
            "Race": "White",
            "Ethnicity": "Not Hispanic",
        }
    row = pd.DataFrame(0, index=[0], columns=X.columns)
    row["Year"]         = new_patient["Year"]
    row["Age_midpoint"] = age_to_midpoint(new_patient["Age"])
    row["Sex_Male"]     = 1 if new_patient["Sex"] == "Male" else 0
    row["Hispanic"]     = 1 if new_patient["Ethnicity"] == "Hispanic" else 0
    race_col = f"Race_{new_patient['Race']}"
    if race_col in row.columns:
        row[race_col] = 1
    row_scaled = scaler.transform(row)

    preds = {}
    for res in results:
        model = res["fitted_model"]
        if res["Model"] == "Linear Regression":
            preds[res["Model"]] = model.predict(row_scaled)[0]
        else:
            preds[res["Model"]] = model.predict(row)[0]
    return new_patient, preds


def plot_new_patient_predictions(new_patient, preds):
    names = list(preds.keys())
    values = list(preds.values())
    colors = ["steelblue", "mediumseagreen", "orange"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, values, color=colors,
                   edgecolor="black", height=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", ha="left",
                fontsize=11, fontweight="bold")
    profile = (f"{new_patient['Year']} | {new_patient['Age']} | "
               f"{new_patient['Sex']} | {new_patient['Race']} | "
               f"{new_patient['Ethnicity']}")
    ax.set_title(f"Predicted Colon Cancer Deaths in CA\n({profile})",
                 fontsize=11)
    ax.set_xlabel("Predicted Deaths")
    ax.set_xlim(0, max(values) * 1.15)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

# Runs CDC WONDER analysis pipeline
def run_full_pipeline(show_plots=True):
    print("Fetching CDC WONDER data...")
    df = fetch_wonder_data()
    print(f"Parsed {len(df)} rows ({df['Suppressed'].sum()} suppressed)")
    df_usable = df[~df["Suppressed"]].copy()
    print(f"Usable rows for ML: {len(df_usable)}")

    if show_plots:
        plot_yearly_totals(df_usable)
        plot_age_boxplot(df_usable)
        plot_sex_race(df_usable)
        plot_ethnicity_trend(df_usable)

    X, y = build_features(df_usable)
    print(f"Feature matrix: {X.shape}")

    results, (_, _, _, y_test, scaler) = train_models(X, y)

    print("\nML Model Summary")
    summary = pd.DataFrame([
        {k: v for k, v in r.items()
         if k not in ("predictions", "fitted_model")}
        for r in results
    ])
    print(summary.to_string(index=False))

    best = max(results, key=lambda r: r["CV_R2_mean"])
    print(f"\nBest model (5-fold CV R^2): {best['Model']}")

    if show_plots:
        plot_predicted_vs_actual(results, y_test)
        plot_feature_importance(results, X)

    new_patient, preds = predict_new_patient(results, X, scaler)
    print("\nNew Patient Prediction")
    for k, v in new_patient.items():
        print(f"  {k}: {v}")
    for name, pred in preds.items():
        print(f"  {name:<20} {pred:>6.1f} deaths")

    if show_plots:
        plot_new_patient_predictions(new_patient, preds)

    return {"df": df, "results": results, "new_patient_preds": preds}


if __name__ == "__main__":
    run_full_pipeline()
