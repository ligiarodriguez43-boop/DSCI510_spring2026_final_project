"""
CRC risk classification on the local CRC CSV dataset.

Trains Linear Regression, Logistic Regression, Random Forest, and Gradient
Boosting models to predict colorectal cancer risk from demographic and
lifestyle features.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, r2_score,
    root_mean_squared_error, roc_auc_score, roc_curve
)

from config import CRC_DATASET_CSV, RANDOM_STATE, TEST_SIZE


FEATURES = ["Age", "BMI", "Gender_Enc", "Lifestyle_Enc",
            "Ethnicity_Enc", "PreCond_Enc"]
FEATURE_NAMES = ["Age", "BMI", "Gender", "Lifestyle",
                 "Ethnicity", "Pre-existing Condition"]
TARGET = "Colorectal_Cancer"

# Load the CRC CSV and rename columns to canonical names
def load_crc_dataset(csv_path=CRC_DATASET_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.rename(columns={
        "Pre-existing Conditions": "PreConditions",
        "CRC_Risk": "Colorectal_Cancer",
    }, inplace=True)
    return df

# Maps a BMI value to a category label
def bmi_category(bmi: float) -> str:
    """Map a BMI value to a category label."""
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"

# Add BMI_Cat, Age_Group, and Risk_Label columns
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BMI_Cat"] = df["BMI"].apply(bmi_category)
    df["Age_Group"] = pd.cut(
        df["Age"], bins=[0, 40, 55, 70, 100],
        labels=["<40", "40-55", "55-70", "70+"]
    )
    df["Risk_Label"] = df["Colorectal_Cancer"].map(
        {0: "No Risk", 1: "Colorectal Cancer"}
    )
    return df

# Label-encode categorical columns. Returns (df, encoders dict)
def encode_features(df: pd.DataFrame):
    df = df.copy()
    encoders = {
        "gender":    LabelEncoder(),
        "lifestyle": LabelEncoder(),
        "ethnicity": LabelEncoder(),
        "precond":   LabelEncoder(),
    }
    df["Gender_Enc"]    = encoders["gender"].fit_transform(df["Gender"])
    df["Lifestyle_Enc"] = encoders["lifestyle"].fit_transform(df["Lifestyle"])
    df["Ethnicity_Enc"] = encoders["ethnicity"].fit_transform(df["Ethnicity"])
    df["PreCond_Enc"]   = encoders["precond"].fit_transform(df["PreConditions"])
    return df, encoders

# Stratified train/test split on the encoded features
def split_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)

# Trains linear regression baseline & returns model and metrics
def train_linear_regression(X_train, y_train, X_test, y_test):
    """Train linear regression baseline; return model and metrics."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, {
        "pred": pred,
        "rmse": root_mean_squared_error(y_test, pred),
        "r2":   r2_score(y_test, pred),
    }

# Trains logistic regression with standardized features
def train_logistic_regression(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                               class_weight="balanced")
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    prob = model.predict_proba(X_test_s)[:, 1]
    return model, scaler, {
        "pred": pred, "prob": prob,
        "auc": roc_auc_score(y_test, prob),
    }


def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100,
                                   random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return model, {
        "pred": pred, "prob": prob,
        "auc": roc_auc_score(y_test, prob),
    }


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                       max_depth=3, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return model, {
        "pred": pred, "prob": prob,
        "auc": roc_auc_score(y_test, prob),
    }

# Predict CRC risk for a single example patient
def predict_new_patient(log_model, scaler, rf_model, gb_model, lr_model,
                        encoders):
    new_patient = pd.DataFrame([{
        "Age":           55,
        "BMI":           28.5,
        "Gender_Enc":    encoders["gender"].transform(["Male"])[0],
        "Lifestyle_Enc": encoders["lifestyle"].transform(["Smoker"])[0],
        "Ethnicity_Enc": encoders["ethnicity"].transform(["Hispanic"])[0],
        "PreCond_Enc":   encoders["precond"].transform(["Diabetes"])[0],
    }])
    new_scaled = scaler.transform(new_patient)
    return {
        "log_pred":  log_model.predict(new_scaled)[0],
        "log_prob":  log_model.predict_proba(new_scaled)[0][1],
        "rf_pred":   rf_model.predict(new_patient)[0],
        "rf_prob":   rf_model.predict_proba(new_patient)[0][1],
        "gb_pred":   gb_model.predict(new_patient)[0],
        "gb_prob":   gb_model.predict_proba(new_patient)[0][1],
        "lr_score":  lr_model.predict(new_patient)[0],
    }


# Data Visualizations
def plot_correlation_heatmap(df: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    cols = FEATURES + [TARGET]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=True, fmt=".2f",
                linewidths=0.5, center=0,
                xticklabels=FEATURE_NAMES + ["Colorectal Cancer"],
                yticklabels=FEATURE_NAMES + ["Colorectal Cancer"])
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_linear_regression_fit(y_test, lr_pred, rmse, r2):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, lr_pred, alpha=0.5, edgecolors="none")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             "r--", lw=2, label="Perfect Fit")
    plt.xlabel("Actual Colorectal Cancer")
    plt.ylabel("Predicted Colorectal Cancer Score")
    plt.title(f"Linear Regression - Actual vs Predicted\n"
              f"RMSE: {rmse:.4f}  |  R2: {r2:.4f}",
              fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_logistic(y_test, log_pred):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, log_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Risk",
                                                      "Colorectal Cancer"])
    disp.plot(ax=plt.gca(), colorbar=False, cmap="Blues")
    plt.title("Confusion Matrix - Logistic Regression",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_age_gender_grouped(df: pd.DataFrame):
    plt.figure(figsize=(9, 6))
    age_gender = df.groupby(["Age_Group", "Gender"], observed=True)[
        "Colorectal_Cancer"].mean() * 100
    age_gender = age_gender.unstack()
    age_groups = age_gender.index.tolist()
    genders = age_gender.columns.tolist()
    x = np.arange(len(age_groups))
    width = 0.35
    for i, gender in enumerate(genders):
        offset = (i - (len(genders) - 1) / 2) * width
        bars = plt.bar(x + offset, age_gender[gender].values, width,
                       label=gender, edgecolor="white")
        for bar, val in zip(bars, age_gender[gender].values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", fontsize=9)
    plt.xticks(x, age_groups)
    plt.xlabel("Age Group")
    plt.ylabel("Colorectal Cancer Rate (%)")
    plt.title("Colorectal Cancer Rate by Age Group and Gender",
              fontsize=13, fontweight="bold")
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.show()


def plot_roc_comparison(y_test, log_prob, rf_prob, gb_prob):
    plt.figure(figsize=(7, 6))
    for name, prob in [("Logistic Regression", log_prob),
                       ("Random Forest", rf_prob),
                       ("Gradient Boosting", gb_prob)]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_v = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_v:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Model Comparison",
              fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rate_by_category(df, column, title, order=None):
    plt.figure(figsize=(8, 5))
    rate = df.groupby(column)["Colorectal_Cancer"].mean()
    if order:
        rate = rate.reindex(order)
    else:
        rate = rate.sort_values(ascending=False)
    rate = rate * 100
    bars = plt.bar(rate.index, rate.values, edgecolor="white")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylabel("Colorectal Cancer Rate (%)")
    plt.xticks(rotation=20)
    for bar, val in zip(bars, rate.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_new_patient_probabilities(log_prob, rf_prob, gb_prob):
    plt.figure(figsize=(7, 5))
    names = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    probs = [log_prob, rf_prob, gb_prob]
    bars = plt.bar(names, [p * 100 for p in probs],
                   edgecolor="white", width=0.5)
    plt.axhline(50, color="red", linestyle="--", lw=1.5,
                label="50% threshold")
    plt.ylabel("Colorectal Cancer Probability (%)")
    plt.ylim(0, 100)
    plt.title("New Patient: Predicted Colorectal Cancer Probability",
              fontsize=13, fontweight="bold")
    plt.legend()
    for bar, val in zip(bars, probs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1%}", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()

# Runs the complete CRC analysis pipeline
def run_full_pipeline(csv_path=CRC_DATASET_CSV, show_plots=True):
    df = load_crc_dataset(csv_path)
    df = add_derived_columns(df)
    print(f"\nDataset: {df.shape[0]} participants | "
          f"Colorectal Cancer positive: {df['Colorectal_Cancer'].sum()} "
          f"({df['Colorectal_Cancer'].mean() * 100:.1f}%)\n")

    df, encoders = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}\n")

    lr_model, lr_metrics = train_linear_regression(
        X_train, y_train, X_test, y_test)
    print(f"Linear Regression  |  RMSE: {lr_metrics['rmse']:.4f}  |  "
          f"R2: {lr_metrics['r2']:.4f}\n")

    log_model, scaler, log_metrics = train_logistic_regression(
        X_train, y_train, X_test, y_test)
    print(f"Logistic Regression  |  AUC: {log_metrics['auc']:.4f}")
    print(classification_report(
        y_test, log_metrics["pred"],
        target_names=["No Risk", "Colorectal Cancer"], zero_division=0))

    rf_model, rf_metrics = train_random_forest(
        X_train, y_train, X_test, y_test)
    print(f"Random Forest  |  AUC: {rf_metrics['auc']:.4f}")
    print(classification_report(
        y_test, rf_metrics["pred"],
        target_names=["No Risk", "Colorectal Cancer"], zero_division=0))

    gb_model, gb_metrics = train_gradient_boosting(
        X_train, y_train, X_test, y_test)
    print(f"Gradient Boosting  |  AUC: {gb_metrics['auc']:.4f}")
    print(classification_report(
        y_test, gb_metrics["pred"],
        target_names=["No Risk", "Colorectal Cancer"], zero_division=0))

    new_pred = predict_new_patient(log_model, scaler, rf_model, gb_model,
                                   lr_model, encoders)
    print("\nNew Patient (Age 55, BMI 28.5, Male, Smoker, Hispanic, Diabetes)")
    print(f"  Logistic Regression -> {new_pred['log_prob']:.2%}")
    print(f"  Random Forest       -> {new_pred['rf_prob']:.2%}")
    print(f"  Gradient Boosting   -> {new_pred['gb_prob']:.2%}")
    print(f"  Linear Regression score -> {new_pred['lr_score']:.4f}")

    if show_plots:
        plot_correlation_heatmap(df)
        plot_linear_regression_fit(y_test, lr_metrics["pred"],
                                   lr_metrics["rmse"], lr_metrics["r2"])
        plot_confusion_matrix_logistic(y_test, log_metrics["pred"])
        plot_age_gender_grouped(df)
        plot_roc_comparison(y_test, log_metrics["prob"],
                            rf_metrics["prob"], gb_metrics["prob"])
        plot_rate_by_category(df, "Lifestyle",
                              "Colorectal Cancer Rate by Lifestyle")
        plot_rate_by_category(df, "PreConditions",
                              "Colorectal Cancer Rate by Pre-existing Condition")
        plot_rate_by_category(df, "BMI_Cat",
                              "Colorectal Cancer Rate by BMI Category",
                              order=["Underweight", "Normal",
                                     "Overweight", "Obese"])
        plot_rate_by_category(df, "Ethnicity",
                              "Colorectal Cancer Rate by Ethnicity")
        plot_new_patient_probabilities(
            new_pred["log_prob"], new_pred["rf_prob"], new_pred["gb_prob"])

    print("\nModel Comparison Summary (AUC)")
    print(f"  Logistic Regression: {log_metrics['auc']:.3f}")
    print(f"  Random Forest:       {rf_metrics['auc']:.3f}")
    print(f"  Gradient Boosting:   {gb_metrics['auc']:.3f}")

    return {
        "df": df,
        "models": {"linear": lr_model, "logistic": log_model,
                   "rf": rf_model, "gb": gb_model},
        "metrics": {"lr": lr_metrics, "log": log_metrics,
                    "rf": rf_metrics, "gb": gb_metrics},
        "new_patient": new_pred,
    }


if __name__ == "__main__":
    run_full_pipeline()
