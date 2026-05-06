"""
Global colorectal cancer mortality classification.

Trains Logistic Regression, Decision Tree, and Gradient Boosting on the
global colorectal cancer dataset to predict 5-year mortality.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, average_precision_score,
)
from sklearn.calibration import calibration_curve

from config import COLORECTAL_CANCER_DATASET_CSV, RANDOM_STATE, TEST_SIZE


MODEL_COLORS = {
    "Logistic Regression": "red",
    "Decision Tree":       "blue",
    "Gradient Boosting":   "green",
}

DROPPED_COLUMNS = {
    "Patient_ID":          "no predictive value",
    "Survival_5_years":    "the same as Mortality",
    "Survival_Prediction": "the same as Mortality",
    "Mortality":           "target variable being predicted",
}

FEATURE_CATEGORIES = {
    "Demographics": {
        "color": "steelblue",
        "features": ["Age", "Gender", "Country", "Urban_or_Rural"],
    },
    "Clinical / Tumor": {
        "color": "darkgoldenrod",
        "features": ["Cancer_Stage", "Tumor_Size_mm"],
    },
    "Risk Factors & Medical History": {
        "color": "purple",
        "features": [
            "Family_History", "Smoking_History", "Alcohol_Consumption",
            "Obesity_BMI", "Diet_Risk", "Physical_Activity",
            "Diabetes", "Inflammatory_Bowel_Disease", "Genetic_Mutation",
        ],
    },
    "Care & Treatment": {
        "color": "green",
        "features": ["Screening_History", "Early_Detection", "Treatment_Type"],
    },
    "Socioeconomic / Healthcare System": {
        "color": "chocolate",
        "features": [
            "Healthcare_Costs", "Incidence_Rate_per_100K",
            "Mortality_Rate_per_100K",
            "Economic_Classification", "Healthcare_Access",
            "Insurance_Status",
        ],
    },
}

# Load the global colorectal cancer CSV."""
def load_global_dataset(csv_path=COLORECTAL_CANCER_DATASET_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path)

# Drop non-feature columns, build target, and one-hot encode
def prepare_features(df: pd.DataFrame):
    df_clean = df.drop(columns=list(DROPPED_COLUMNS.keys()))
    y = (df["Mortality"] == "Yes").astype(int)
    X = df_clean

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return {
        "X_encoded": X_encoded,
        "y": y,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "raw_feature_names": X.columns.tolist(),
    }

# Trains Logistic Regression, Decision Tree, Gradient Boosting
def train_models(prepared):
    X_encoded = prepared["X_encoded"]
    y = prepared["y"]
    numeric_cols = prepared["numeric_cols"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)

    dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=50,
                                random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)

    gb = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                    learning_rate=0.1,
                                    random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)

    models = {
        "Logistic Regression": {"clf": lr, "X_test": X_test_scaled},
        "Decision Tree":       {"clf": dt, "X_test": X_test},
        "Gradient Boosting":   {"clf": gb, "X_test": X_test},
    }
    for name, m in models.items():
        m["y_pred"]  = m["clf"].predict(m["X_test"])
        m["y_proba"] = m["clf"].predict_proba(m["X_test"])[:, 1]
        m["accuracy"]      = accuracy_score(y_test, m["y_pred"])
        m["precision"]     = precision_score(y_test, m["y_pred"], zero_division=0)
        m["recall"]        = recall_score(y_test, m["y_pred"])
        m["f1"]            = f1_score(y_test, m["y_pred"])
        m["roc_auc"]       = roc_auc_score(y_test, m["y_proba"])
        m["avg_precision"] = average_precision_score(y_test, m["y_proba"])

    return {
        "models": models,
        "X_train": X_train, "X_test": X_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler,
    }


def performance_summary(models) -> pd.DataFrame:
    return pd.DataFrame(
        {name: {k: m[k] for k in ["accuracy", "precision", "recall",
                                  "f1", "roc_auc", "avg_precision"]}
         for name, m in models.items()}
    ).T.round(3)


def print_confusion_matrices(models, y_test):
    print("CONFUSION MATRICES")
    for name, m in models.items():
        cm = confusion_matrix(y_test, m["y_pred"])
        print(f"\n{name}:")
        print(f"  Survived (actual): {cm[0,0]:>6,} pred survived | "
              f"{cm[0,1]:>6,} pred died")
        print(f"  Died     (actual): {cm[1,0]:>6,} pred survived | "
              f"{cm[1,1]:>6,} pred died")


def plot_roc_curves(models, y_test):
    plt.figure(figsize=(8, 6))
    for name, m in models.items():
        fpr, tpr, _ = roc_curve(y_test, m["y_proba"])
        plt.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2,
                 label=f"{name} (AUC = {m['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--",
             linewidth=1, label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for All Three Models")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_calibration(models, y_test):
    plt.figure(figsize=(8, 6))
    for name, m in models.items():
        frac_pos, mean_pred = calibration_curve(
            y_test, m["y_proba"], n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o",
                 color=MODEL_COLORS[name], label=name)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Actual fraction of positives")
    plt.title("Calibration Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Predict survival for a new patient profile
def predict_new_patient(trained, prepared, new_patient_dict=None):
    if new_patient_dict is None:
        new_patient_dict = {
            "Age": 58, "Gender": "Male", "Country": "USA",
            "Urban_or_Rural": "Urban", "Cancer_Stage": "Regional",
            "Tumor_Size_mm": 45, "Family_History": "Yes",
            "Smoking_History": "Yes", "Alcohol_Consumption": "No",
            "Obesity_BMI": "Obese", "Diet_Risk": "High",
            "Physical_Activity": "Low", "Diabetes": "No",
            "Inflammatory_Bowel_Disease": "No", "Genetic_Mutation": "No",
            "Screening_History": "Regular", "Early_Detection": "Yes",
            "Treatment_Type": "Surgery", "Healthcare_Costs": 45000,
            "Incidence_Rate_per_100K": 40, "Mortality_Rate_per_100K": 15,
            "Economic_Classification": "High", "Healthcare_Access": "Good",
            "Insurance_Status": "Insured",
        }
    raw = pd.DataFrame([new_patient_dict])
    encoded = pd.get_dummies(raw, columns=prepared["categorical_cols"],
                             drop_first=True)
    encoded = encoded.reindex(columns=prepared["X_encoded"].columns,
                              fill_value=0)
    scaled = encoded.copy()
    scaled[prepared["numeric_cols"]] = trained["scaler"].transform(
        encoded[prepared["numeric_cols"]])

    lr = trained["models"]["Logistic Regression"]["clf"]
    dt = trained["models"]["Decision Tree"]["clf"]
    gb = trained["models"]["Gradient Boosting"]["clf"]

    return {
        "Logistic Regression": {
            "pred": lr.predict(scaled)[0],
            "death_prob": lr.predict_proba(scaled)[0][1],
        },
        "Decision Tree": {
            "pred": dt.predict(encoded)[0],
            "death_prob": dt.predict_proba(encoded)[0][1],
        },
        "Gradient Boosting": {
            "pred": gb.predict(encoded)[0],
            "death_prob": gb.predict_proba(encoded)[0][1],
        },
    }

# global colorectal cancer mortality classification piptline
def run_full_pipeline(csv_path=COLORECTAL_CANCER_DATASET_CSV, show_plots=True):
    print("Loading global colorectal cancer dataset...")
    df = load_global_dataset(csv_path)
    print(f"Shape: {df.shape}")

    prepared = prepare_features(df)
    print(f"Features after one-hot encoding: {prepared['X_encoded'].shape[1]}")

    trained = train_models(prepared)
    print(f"Train: {len(trained['X_train'])} | Test: {len(trained['X_test'])}\n")

    summary = performance_summary(trained["models"])
    print("Model Performance Summary")
    print(summary.to_string())

    print_confusion_matrices(trained["models"], trained["y_test"])

    if show_plots:
        plot_roc_curves(trained["models"], trained["y_test"])
        plot_calibration(trained["models"], trained["y_test"])

    new_pred = predict_new_patient(trained, prepared)
    print("\nNew Patient Prediction (Age 58, Male, Stage Regional)")
    for name, r in new_pred.items():
        verdict = "Did Not Survive" if r["pred"] == 1 else "Survived"
        print(f"  {name:<22} -> {verdict}  "
              f"(survival prob: {1 - r['death_prob']:.2%})")

    return {"df": df, "prepared": prepared, "trained": trained,
            "new_patient": new_pred}


if __name__ == "__main__":
    run_full_pipeline()
