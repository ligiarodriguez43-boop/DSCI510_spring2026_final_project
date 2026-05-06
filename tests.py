"""
Unit tests for the CRC analysis project.

Run with:
    python -m pytest tests.py -v
    # or
    python tests.py
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).parent / "src"))

from crc_analysis import (
    bmi_category, add_derived_columns, encode_features,
)
from cdc_wonder import age_to_midpoint, parse_wonder_response
from global_rankings import prepare_features, DROPPED_COLUMNS
import config


class TestConfig(unittest.TestCase):

    def test_paths_resolve(self):
        self.assertTrue(config.SRC_DIR.exists())
        self.assertTrue(config.PROJECT_DIR.exists())

    def test_random_state_is_int(self):
        self.assertIsInstance(config.RANDOM_STATE, int)

    def test_test_size_in_range(self):
        self.assertGreater(config.TEST_SIZE, 0)
        self.assertLess(config.TEST_SIZE, 1)

    def test_get_nci_api_key_raises_without_env(self):
        original = os.environ.pop(config.ENV_NCI_API_KEY, None)
        try:
            with self.assertRaises(RuntimeError):
                config.get_nci_api_key()
        finally:
            if original is not None:
                os.environ[config.ENV_NCI_API_KEY] = original


class TestCRCAnalysisHelpers(unittest.TestCase):
    def test_bmi_category_underweight(self):
        self.assertEqual(bmi_category(17.0), "Underweight")

    def test_bmi_category_normal(self):
        self.assertEqual(bmi_category(22.0), "Normal")

    def test_bmi_category_overweight(self):
        self.assertEqual(bmi_category(27.5), "Overweight")

    def test_bmi_category_obese(self):
        self.assertEqual(bmi_category(31.0), "Obese")

    def test_bmi_category_boundary_18_5(self):
        # Boundary: 18.5 exactly is Normal, not Underweight
        self.assertEqual(bmi_category(18.5), "Normal")

    def test_add_derived_columns(self):
        df = pd.DataFrame({
            "Age": [30, 50, 65, 80],
            "BMI": [20.0, 28.0, 31.0, 17.0],
            "Colorectal_Cancer": [0, 1, 0, 1],
        })
        out = add_derived_columns(df)
        self.assertIn("BMI_Cat", out.columns)
        self.assertIn("Age_Group", out.columns)
        self.assertIn("Risk_Label", out.columns)
        self.assertEqual(out["BMI_Cat"].tolist(),
                         ["Normal", "Overweight", "Obese", "Underweight"])
        self.assertEqual(out["Risk_Label"].tolist(),
                         ["No Risk", "Colorectal Cancer",
                          "No Risk", "Colorectal Cancer"])

    def test_encode_features_produces_int_columns(self):
        df = pd.DataFrame({
            "Age": [30, 50],
            "BMI": [20.0, 28.0],
            "Gender": ["Male", "Female"],
            "Lifestyle": ["Smoker", "Active"],
            "Ethnicity": ["Hispanic", "Asian"],
            "PreConditions": ["None", "Diabetes"],
            "Colorectal_Cancer": [0, 1],
        })
        out, encoders = encode_features(df)
        for col in ["Gender_Enc", "Lifestyle_Enc",
                    "Ethnicity_Enc", "PreCond_Enc"]:
            self.assertIn(col, out.columns)
            self.assertTrue(np.issubdtype(out[col].dtype, np.integer))
        self.assertEqual(set(encoders.keys()),
                         {"gender", "lifestyle", "ethnicity", "precond"})

# Tests functions in cdc_wonder
class TestCDCWonderHelpers(unittest.TestCase):

    def test_age_to_midpoint_range(self):
        self.assertAlmostEqual(age_to_midpoint("65-69 years"), 67.0)

    def test_age_to_midpoint_85_plus(self):
        self.assertAlmostEqual(age_to_midpoint("85+ years"), 87.0)

    def test_age_to_midpoint_unknown_returns_nan(self):
        self.assertTrue(np.isnan(age_to_midpoint("Unknown")))

    def test_age_to_midpoint_strips_whitespace(self):
        self.assertAlmostEqual(age_to_midpoint("  50-54 years  "), 52.0)

    def test_parse_wonder_response_raises_on_missing_table(self):
        with self.assertRaises(RuntimeError):
            parse_wonder_response("<html><body>No data here</body></html>")


class TestGlobalRankings(unittest.TestCase):
    def test_prepare_features_drops_target_columns(self):
        df = pd.DataFrame({
            "Patient_ID": [1, 2, 3, 4],
            "Age": [50, 60, 70, 45],
            "Gender": ["Male", "Female", "Male", "Female"],
            "Country": ["USA", "UK", "USA", "UK"],
            "Urban_or_Rural": ["Urban", "Rural", "Urban", "Rural"],
            "Cancer_Stage": ["Localized", "Regional",
                             "Metastatic", "Localized"],
            "Tumor_Size_mm": [20, 40, 60, 15],
            "Family_History": ["Yes", "No", "Yes", "No"],
            "Smoking_History": ["No", "Yes", "Yes", "No"],
            "Alcohol_Consumption": ["No", "Yes", "No", "Yes"],
            "Obesity_BMI": ["Normal", "Obese", "Overweight", "Normal"],
            "Diet_Risk": ["Low", "High", "Medium", "Low"],
            "Physical_Activity": ["High", "Low", "Medium", "High"],
            "Diabetes": ["No", "Yes", "No", "No"],
            "Inflammatory_Bowel_Disease": ["No", "No", "Yes", "No"],
            "Genetic_Mutation": ["No", "No", "Yes", "No"],
            "Screening_History": ["Regular", "None",
                                  "Irregular", "Regular"],
            "Early_Detection": ["Yes", "No", "No", "Yes"],
            "Treatment_Type": ["Surgery", "Chemo",
                               "Radiation", "Surgery"],
            "Healthcare_Costs": [10000, 20000, 30000, 5000],
            "Incidence_Rate_per_100K": [40, 35, 50, 30],
            "Mortality_Rate_per_100K": [15, 20, 25, 10],
            "Economic_Classification": ["High", "Mid", "High", "Mid"],
            "Healthcare_Access": ["Good", "Poor", "Good", "Good"],
            "Insurance_Status": ["Insured", "Uninsured",
                                 "Insured", "Insured"],
            "Survival_5_years": ["Yes", "No", "No", "Yes"],
            "Survival_Prediction": ["Yes", "No", "No", "Yes"],
            "Mortality": ["No", "Yes", "Yes", "No"],
        })
        prepared = prepare_features(df)
        for col in DROPPED_COLUMNS:
            self.assertNotIn(col, prepared["X_encoded"].columns,
                             f"{col} should be dropped")
        # Target should be 0/1
        self.assertEqual(set(prepared["y"].unique()), {0, 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
