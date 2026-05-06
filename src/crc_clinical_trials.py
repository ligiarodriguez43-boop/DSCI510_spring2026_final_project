"""
NCI Clinical Trials API: fetch colon cancer trials and predict whether
a trial reports survival outcomes from its treatment composition.
"""
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, ConfusionMatrixDisplay)

from config import (NCI_TRIALS_BASE_URL, TARGET_TRIALS, TRIALS_PAGE_SIZE,
                    TRIALS_OUTPUT_JSON, RANDOM_STATE, get_nci_api_key)


HEATMAP_COLS = ["has_immunotherapy", "has_targeted",
                "has_chemo", "has_radiation"]
TREATMENT_LABELS = ["Immunotherapy", "Targeted", "Chemo", "Radiation"]


def _headers():
    return {"X-API-KEY": get_nci_api_key(),
            "Content-Type": "application/json"}


def _get_first(d, *keys):
    for k in keys:
        v = d.get(k)
        if v:
            return v
    return None

# Look up NCI thesaurus disease codes that matches a keyword
def fetch_disease_codes(keyword="colon", size=400):
    response = requests.get(
        f"{NCI_TRIALS_BASE_URL}/diseases",
        headers=_headers(),
        params={"name": keyword, "size": size},
    ).json()
    diseases = (response.get("terms") or response.get("data")
                or response.get("results") or response.get("diseases") or [])

    candidate_codes = []
    for d in diseases:
        name = (_get_first(d, "name", "preferred_name", "display_name") or "").lower()
        if "colon" in name and any(kw in name for kw in [
                "cancer", "carcinoma", "neoplasm", "adenocarcinoma",
                "tumor", "malignant"]):
            if "rectal" in name and "colon" not in name.split("rectal")[0]:
                continue
            code = _get_first(d, "codes", "nci_thesaurus_concept_id",
                              "concept_id", "code")
            if isinstance(code, list):
                candidate_codes.extend(code)
            elif code:
                candidate_codes.append(code)

    return list(dict.fromkeys(candidate_codes))

# Fetch trials by disease codes, with keyword fallback
def fetch_trials(disease_codes, target=TARGET_TRIALS):
    """Fetch trials by disease codes, with keyword fallback."""
    trials = []
    seen_nct = set()

    def add_batch(batch):
        added = 0
        for t in batch:
            nct = t.get("nct_id")
            if nct and nct not in seen_nct:
                seen_nct.add(nct)
                trials.append(t)
                added += 1
        return added

    top_codes = disease_codes[:20]

    # POST with body
    if top_codes:
        from_index = 0
        while len(trials) < target:
            body = {
                "diseases.nci_thesaurus_concept_id": top_codes,
                "size": TRIALS_PAGE_SIZE,
                "from": from_index,
            }
            data = requests.post(f"{NCI_TRIALS_BASE_URL}/trials",
                                 headers=_headers(), json=body).json()
            batch = data.get("trials") or data.get("data") or []
            if not batch:
                break
            if add_batch(batch) == 0:
                break
            from_index += TRIALS_PAGE_SIZE

    # GET per code
    if len(trials) < target and top_codes:
        for code in top_codes:
            if len(trials) >= target:
                break
            from_index = 0
            while len(trials) < target:
                data = requests.get(f"{NCI_TRIALS_BASE_URL}/trials",
                                    headers=_headers(),
                                    params={"diseases.nci_thesaurus_concept_id": code,
                                            "size": TRIALS_PAGE_SIZE,
                                            "from": from_index}).json()
                batch = data.get("trials") or data.get("data") or []
                if not batch:
                    break
                if add_batch(batch) == 0:
                    break
                from_index += TRIALS_PAGE_SIZE

    # keyword fallback
    if len(trials) < target:
        from_index = 0
        while len(trials) < target and from_index < 2000:
            data = requests.get(f"{NCI_TRIALS_BASE_URL}/trials",
                                headers=_headers(),
                                params={"keyword": "colon cancer",
                                        "size": TRIALS_PAGE_SIZE,
                                        "from": from_index}).json()
            batch = data.get("trials") or []
            if not batch:
                break
            kept = []
            for t in batch:
                title = (t.get("brief_title") or "").lower()
                disease_blob = ""
                for d in (t.get("diseases") or []):
                    for k, v in d.items():
                        if isinstance(v, str) and "name" in k.lower():
                            disease_blob += " " + v.lower()
                if "colon" in title or "colon" in disease_blob:
                    kept.append(t)
            add_batch(kept)
            from_index += TRIALS_PAGE_SIZE

    return trials

# Extracts a flat row of features from one trial record
def extract_survival_fields(trial):
    arms = trial.get("arms", [])
    all_interventions = []
    for arm in arms:
        for i in arm.get("interventions", []):
            name = i.get("name", "") or i.get("intervention_name", "")
            all_interventions.append(name.lower())
    drugs_str = " ".join(all_interventions)

    outcomes = (trial.get("outcome_measures", [])
                or trial.get("primary_outcomes", []))
    survival_outcomes = [
        o for o in outcomes
        if any(kw in json.dumps(o).lower() for kw in [
            "overall survival", "progression-free", "disease-free",
            "mortality", "survival rate", "os ", "pfs"])
    ]

    eligibility = trial.get("eligibility", {}).get("structured", {})
    disease_names = []
    for d in (trial.get("diseases") or []):
        for k, v in d.items():
            if isinstance(v, str) and "name" in k.lower():
                disease_names.append(v)

    return {
        "nct_id":            trial.get("nct_id"),
        "title":             trial.get("brief_title"),
        "phase":             trial.get("phase"),
        "status":            trial.get("current_trial_status"),
        "disease_names":     list(set(disease_names))[:5],
        "drugs":             all_interventions,
        "num_drugs":         len(all_interventions),
        "survival_outcomes": survival_outcomes,
        "has_survival_data": len(survival_outcomes) > 0,
        "min_age":           eligibility.get("min_age_in_years"),
        "max_age":           eligibility.get("max_age_in_years"),
        "has_immunotherapy": any(kw in drugs_str for kw in [
            "pembrolizumab", "nivolumab", "atezolizumab",
            "ipilimumab", "durvalumab", "immune"]),
        "has_targeted":      any(kw in drugs_str for kw in [
            "bevacizumab", "cetuximab", "panitumumab",
            "regorafenib", "ramucirumab"]),
        "has_chemo":         any(kw in drugs_str for kw in [
            "folfox", "folfiri", "oxaliplatin", "irinotecan",
            "fluorouracil", "capecitabine", "leucovorin",
            "5-fu", "xeloda"]),
        "has_radiation":     any(kw in drugs_str for kw in [
            "radiation", "radiotherapy", "xrt"]),
    }

# Extract features for every trial and (optionally) save to JSON."""
def build_trials_dataframe(trials, save_to=TRIALS_OUTPUT_JSON):
    extracted = [extract_survival_fields(t) for t in trials]
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        with open(save_to, "w") as f:
            json.dump(extracted, f, indent=2)
    return pd.DataFrame(extracted)


# Data  Visualizations
def plot_treatment_vs_survival(df):
    heatmap_data = df.groupby("has_survival_data")[
        HEATMAP_COLS].sum().fillna(0).astype(int)
    no_surv = (heatmap_data.loc[False].values
               if False in heatmap_data.index else [0] * 4)
    has_surv = (heatmap_data.loc[True].values
                if True in heatmap_data.index else [0] * 4)
    x = list(range(len(TREATMENT_LABELS)))
    width = 0.35

    plt.figure(figsize=(10, 6))
    b1 = plt.bar([i - width / 2 for i in x], no_surv, width,
                 label="No Survival Data", color="red")
    b2 = plt.bar([i + width / 2 for i in x], has_surv, width,
                 label="Has Survival Data", color="green")
    for bar in list(b1) + list(b2):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 str(int(bar.get_height())), ha="center", fontsize=10)
    plt.title(f"Treatment Types vs Survival Outcome Availability (n={len(df)})")
    plt.ylabel("Number of Trials")
    plt.xticks(x, TREATMENT_LABELS)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_survival_donut(df):
    plt.figure(figsize=(7, 7))
    counts = df["has_survival_data"].value_counts()
    plt.pie(counts.values,
            labels=["Has Survival Data" if i else "No Survival Data"
                    for i in counts.index],
            colors=["green" if i else "red" for i in counts.index],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"width": 0.4, "edgecolor": "white"})
    plt.title("Proportion of Colon Cancer Trials Reporting Survival Outcomes")
    plt.tight_layout()
    plt.show()


def plot_phase_treatment_stacked(df):
    pdf = df.copy()
    pdf["phase"] = pdf["phase"].fillna("Unknown")
    phase_treat = pdf.groupby("phase")[HEATMAP_COLS].sum()
    phase_treat.columns = TREATMENT_LABELS
    phase_treat.plot(kind="bar", stacked=True, figsize=(11, 6),
                     colormap="viridis", edgecolor="white")
    plt.title("Treatment Modalities by Cancer Trial Phase")
    plt.ylabel("Number of Trials")
    plt.xlabel("Trial Phase")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Treatment Type")
    plt.tight_layout()
    plt.show()


def plot_treatment_cooccurrence(df):
    co = pd.DataFrame(index=TREATMENT_LABELS,
                      columns=TREATMENT_LABELS, dtype=int)
    matrix = df[HEATMAP_COLS].astype(int).values
    for i in range(len(HEATMAP_COLS)):
        for j in range(len(HEATMAP_COLS)):
            co.iloc[i, j] = int(((matrix[:, i] == 1) &
                                 (matrix[:, j] == 1)).sum())
    plt.figure(figsize=(8, 6))
    sns.heatmap(co.astype(int), annot=True, fmt="d",
                cmap="YlOrRd", cbar_kws={"label": "# of Trials"})
    plt.title("Treatment Co-Occurrence Across Trials")
    plt.tight_layout()
    plt.show()


def train_survival_models(df):
    """Train three classifiers to predict survival-data availability."""
    ml_df = df.copy()
    ml_df["phase"] = ml_df["phase"].fillna("Unknown")
    phase_dummies = pd.get_dummies(ml_df["phase"], prefix="phase")

    feature_cols = ["num_drugs"] + HEATMAP_COLS
    X = pd.concat([ml_df[feature_cols].astype(int),
                   phase_dummies.astype(int)], axis=1)
    y = ml_df["has_survival_data"].astype(int)

    if y.nunique() < 2:
        return None  # Can't be trained with one class

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                class_weight="balanced")
    rf.fit(X_train, y_train)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                    learning_rate=0.1, random_state=RANDOM_STATE)
    gb.fit(X_train, y_train, sample_weight=sample_weights)

    return {
        "X_test": X_test, "y_test": y_test,
        "log": log_reg, "rf": rf, "gb": gb,
        "log_pred": log_reg.predict(X_test),
        "log_proba": log_reg.predict_proba(X_test)[:, 1],
        "rf_pred":  rf.predict(X_test),
        "rf_proba": rf.predict_proba(X_test)[:, 1],
        "gb_pred":  gb.predict(X_test),
        "gb_proba": gb.predict_proba(X_test)[:, 1],
    }


def plot_confusion_matrices(results):
    for name, key, cmap in [("Logistic Regression", "log_pred", "Blues"),
                            ("Random Forest",       "rf_pred",  "Greens"),
                            ("Gradient Boosting",   "gb_pred",  "Purples")]:
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(
            confusion_matrix(results["y_test"], results[key]),
            display_labels=["No Survival", "Survival"]
        ).plot(ax=ax, cmap=cmap, colorbar=True)
        ax.set_title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.show()


def plot_roc_curves(results):
    plt.figure(figsize=(8, 6))
    for name, key, color in [
            ("Logistic Regression", "log_proba", "steelblue"),
            ("Random Forest",       "rf_proba",  "darkorange"),
            ("Gradient Boosting",   "gb_proba",  "purple")]:
        fpr, tpr, _ = roc_curve(results["y_test"], results[key])
        plt.plot(fpr, tpr,
                 label=f"{name} (AUC = {auc(fpr, tpr):.2f})",
                 color=color, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray",
             label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Predicting Survival Data Availability")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

# NCI clinical trials analysis pipeline
def run_full_pipeline(show_plots=True):
    print("Fetching colon cancer disease codes...")
    codes = fetch_disease_codes()
    print(f"  Found {len(codes)} disease codes")

    print(f"\nFetching up to {TARGET_TRIALS} trials...")
    trials = fetch_trials(codes)
    print(f"Total trials collected: {len(trials)}")
    if not trials:
        raise SystemExit("Could not fetch any trials.")

    df = build_trials_dataframe(trials)
    print(f"\nFinal dataset: {len(df)} trials. "
          f"{df['has_survival_data'].sum()} report survival outcomes.")

    if show_plots:
        plot_treatment_vs_survival(df)
        plot_survival_donut(df)
        plot_phase_treatment_stacked(df)
        plot_treatment_cooccurrence(df)

    print("\nML: predicting whether a trial reports survival data")
    results = train_survival_models(df)
    if results is None:
        print("  Skipped: target has only one class.")
        return {"df": df, "ml": None}

    print("\nLogistic Regression")
    print(classification_report(results["y_test"], results["log_pred"],
                                target_names=["No Survival", "Has Survival"]))
    print("\nRandom Forest")
    print(classification_report(results["y_test"], results["rf_pred"],
                                target_names=["No Survival", "Has Survival"]))
    print("\nGradient Boosting")
    print(classification_report(results["y_test"], results["gb_pred"],
                                target_names=["No Survival", "Has Survival"]))

    if show_plots:
        plot_confusion_matrices(results)
        plot_roc_curves(results)

    return {"df": df, "ml": results}


if __name__ == "__main__":
    run_full_pipeline()
