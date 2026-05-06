## Title: Colon Cancer Survival Prediction through Machine Learning Models

## Introduction

Colon cancer was once considered a disease mainly affecting middle-aged or older adults, but recent studies have shown a rise in colon cancer among younger people. To address this change, building various machine learning (ML) prediction models can be helpful to estimate mortality rates. The models use datasets containing patient age, global colon cancer data, cancer stage data, risk factors, gender, and survival outcomes. After cleaning and organizing the data, these factors are used to train ML models such as logistic regression, decision trees, and gradient boosting to predict mortality. The models are then evaluated to assess how accurately they predict mortality rates across age groups, genders, colon cancer clinical trials and countries. 

This project draws on four data sources: two Kaggle CSV datasets, the NCI Clinical Trials API, and the CDC WONDER API. For each source, the pipeline performs exploratory data analysis, trains multiple machine learning models (linear/logistic regression, decision tree, random forest, gradient boosting), evaluates them with appropriate metrics (RMSE, R², AUC, classification report, calibration), and produces a worked prediction for an example patient or demographic profile.

## Data sources

| # | Name/Source | URL | Type | List of Fields | Format | Estimated data size |
|---|-------------|-----|------|---------------|--------|---------------------|
| 1 | NIH Colorectal Cancer Clinical Trials | https://clinicaltrialsapi.cancer.gov/api/v2 | API | Healthcare | JSON | 300+ trials |
| 2 | Colorectal Cancer Global Dataset | https://www.kaggle.com/datasets/ankushpanday2/colorectal-cancer-global-dataset-and-predictions/discussion | File | Healthcare | CSV | 167,497 records |
| 3 | Colorectal Cancer Dietary and Lifestyle Dataset – colon cancer risk factors | https://www.kaggle.com/datasets/ziya07/colorectal-cancer-dietary-and-lifestyle-dataset | File | Healthcare | CSV | 1,000 records |
| 4 | CDC Wonder | https://wonder.cdc.gov/wonder/help/wonder-api.html | Web | Healthcare | XML | 300+ rows |

## Analysis

The project performs four parallel analyses, each combining EDA, multiple ML models, and a worked prediction:

**1. Colon cancer risk classification** (`src/crc_analysis.py`) — uses the dietary and lifestyle Kaggle dataset to predict a binary CRC risk label from demographic and lifestyle features. Compares Linear Regression (as a calibration baseline), Logistic Regression with class-weight balancing, Random Forest, and Gradient Boosting. Evaluated with classification report and ROC-AUC.

**2. Global mortality (survival) classification** (`src/global_rankings.py`) — uses the global Kaggle dataset (~167K records) to predict 5 year mortality. One-hot encodes categoricals, scales inputs for Logistic Regression, and adds calibration curve analysis to assess probability reliability. Compares Logistic Regression, Decision Tree, and Gradient Boosting.

**3. Clinical trials survival reporting** (`src/crc_clinical_trials.py`) — fetches up to 400 colon cancer trials from the NCI Clinical Trials API via three fallback strategies (POST with disease codes, GET per code, keyword fallback). Extracts treatment-modality features (immunotherapy / targeted therapy / chemo / radiation) and trial phase, then predicts whether a trial reports survival outcomes.

**4. Population mortality regression** (`src/cdc_wonder.py`) — POSTs an XML query to CDC WONDER for California colon cancer deaths 2018–2022, parses the row-span-encoded HTML response, encodes age ranges as numeric midpoints, and trains regressors with 5-fold cross-validation. Compares Linear Regression, Random Forest, and Gradient Boosting on R², MAE, and RMSE.

Across all four, plots include correlation heatmaps, ROC curves, confusion matrices, feature importance bars, and a horizontal bar chart of the new-patient (or new demographic profile) prediction across the trained models.

## Summary of the results

- **Local CRC classifier:** all three classifiers cluster around AUC ≈ 0.50–0.55, suggesting the demographic/lifestyle features alone provide limited discriminative signal — consistent with CRC being multifactorial. Linear regression baseline reports near-zero R², reinforcing this.
- **Global mortality classifier:** Logistic Regression and Gradient Boosting both reach AUC in the mid-0.50s, with the Decision Tree underperforming. Calibration analysis shows predicted probabilities are reasonably well-aligned with observed mortality rates in the middle bins.
- **Clinical trials:** roughly half the fetched colon cancer trials report some survival measure. Phase III/IV trials over-index on survival reporting; the model's strongest predictor is trial phase, with treatment modality contributing modestly.
- **CDC WONDER mortality regression:** Random Forest and Gradient Boosting reach R² around 0.85–0.90, vastly outperforming Linear Regression (R² ≈ 0.65). Age midpoint dominates feature importance, with sex and race contributing meaningful but smaller signal.
  
For a 65–69 year-old white non-Hispanic male in California (2022), the trained models predict deaths in this stratum on the order of dozens per year & it was consistent with the state vital-statistics totals.

## How to run

### 1. Install

```bash
git clone <your-repo-url>
cd crc-analysis
python -m venv venv
source venv/bin/activate         # macOS/Linux
# venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Provide the data

Per the submission guidelines, **no data files are committed to this repo**. To reproduce the local-CSV analyses, download these two Kaggle datasets and place them in `data/`:

**`data/crc_dataset.csv`** — from the Colorectal Cancer Dietary and Lifestyle Dataset (~1,000 records):
- Download: https://www.kaggle.com/datasets/ziya07/colorectal-cancer-dietary-and-lifestyle-dataset
- After download, rename the CSV inside the archive to `crc_dataset.csv` and place it in `data/`.
- Expected columns: `Age`, `BMI`, `Gender`, `Lifestyle`, `Ethnicity`, `Pre-existing Conditions`, `CRC_Risk`

**`data/colorectal_cancer_dataset.csv`** — from the Colorectal Cancer Global Dataset (~167,497 records):
- Download: https://www.kaggle.com/datasets/ankushpanday2/colorectal-cancer-global-dataset-and-predictions
- After download, rename the CSV to `colorectal_cancer_dataset.csv` and place it in `data/`.
- Expected columns include: `Patient_ID`, `Age`, `Gender`, `Country`, `Cancer_Stage`, `Tumor_Size_mm`, `Treatment_Type`, `Mortality` (Yes/No), and others.

The two API analyses (CDC WONDER, NCI Clinical Trials) fetch their data live and need only network access (and an API key for NCI — see below). If you don't have the CSVs and only want to run the API stages, use:

```bash
python main.py --only wonder    # CDC WONDER, no key required
python main.py --only trials    # NCI Clinical Trials, requires NCI_API_KEY
```

### 3. Provide API keys

Copy `.env.example` to `.env` and fill in your NCI API key:

```bash
cp .env.example .env
# Edit .env and paste your key:
# NCI_API_KEY=your_key_here
```

You can request a free NCI Clinical Trials API key at https://clinicaltrialsapi.cancer.gov/.

The CDC WONDER endpoint does not require a key.

### 4. Run the pipeline

```bash
# Runs 
python main.py

# Runs without showing plots 
python main.py --no-plots

# Skip the API-based analyses (no network / no key needed)
python main.py --skip-api

# Runs a single analysis
python main.py --only crc       # local CRC dataset
python main.py --only global    # global rankings
python main.py --only trials    # NCI clinical trials API
python main.py --only wonder    # CDC WONDER API
```

**Note on the NCI Clinical Trials stage:** the NCI API requires a key and is occasionally rate-limited or has schema changes. Each pipeline stage is wrapped in a safe runner & if the trials stage fails, the rest of the pipeline (`crc`, `global`, `wonder`) continues normally and you'll see "FAILED" next to that stage in the summary. To skip API stages entirely, use `--skip-api`.

### 5. Run the notebook

```bash
jupyter notebook results.ipynb
```

The notebook calls the same module functions as `main.py` — there is no duplicated code.

### 6. Run the tests

```bash
python tests.py
# or
python -m pytest tests.py -v
```

The test suite includes 17 tests covering config validation, data-loading and feature-engineering helpers (BMI categorization, age-range parsing, encoding), the WONDER HTML parser's error path, and feature preparation for the global dataset.

