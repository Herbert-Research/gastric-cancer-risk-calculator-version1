"""End-to-end integration tests for the gastric cancer risk calculator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from risk_calculator import (
    GastricCancerRiskModel,
    load_model_config,
    load_tcga_cohort,
    predict_with_both_models,
    load_survival_model,
    score_patients,
    sigmoid,
    resolve_ln_ratio,
    normalize_t_stage,
    normalize_n_stage,
    estimate_tumor_size,
    estimate_ln_ratio,
    parse_event_status,
    safe_float,
)


class TestSigmoidFunction:
    """Test the numerically stable sigmoid implementation."""

    def test_sigmoid_zero(self):
        """Test sigmoid(0) = 0.5."""
        assert sigmoid(0.0) == 0.5

    def test_sigmoid_positive(self):
        """Test sigmoid with positive values."""
        result = sigmoid(2.0)
        assert 0.8 < result < 0.9

    def test_sigmoid_negative(self):
        """Test sigmoid with negative values."""
        result = sigmoid(-2.0)
        assert 0.1 < result < 0.2

    def test_sigmoid_large_negative_no_overflow(self):
        """Test sigmoid handles large negative values without overflow."""
        result = sigmoid(-710)
        assert result == 0.0 or result > 0.0  # Should not raise

    def test_sigmoid_large_positive(self):
        """Test sigmoid handles large positive values."""
        result = sigmoid(710)
        assert result < 1.0 or result == 1.0  # Should not raise


class TestSafeFloat:
    """Test the safe_float utility function."""

    def test_safe_float_python_number(self):
        """Test with standard Python numbers."""
        assert safe_float(3.14) == 3.14
        assert safe_float(42) == 42.0

    def test_safe_float_numpy_scalar(self):
        """Test with numpy scalar types."""
        import numpy as np

        assert safe_float(np.float64(3.14)) == 3.14
        assert safe_float(np.int32(42)) == 42.0


class TestResolveLnRatio:
    """Test the LN ratio resolution logic."""

    def test_direct_ratio_used(self):
        """Test that direct ratio is used when provided."""
        assert resolve_ln_ratio(0.25, None, None) == 0.25

    def test_calculated_from_counts(self):
        """Test ratio is calculated from counts when direct not provided."""
        assert resolve_ln_ratio(None, 5, 20) == 0.25

    def test_returns_none_for_zero_total(self):
        """Test returns None when total LN is 0."""
        assert resolve_ln_ratio(None, 0, 0) is None

    def test_returns_none_for_missing_data(self):
        """Test returns None when insufficient data."""
        assert resolve_ln_ratio(None, None, None) is None
        assert resolve_ln_ratio(None, 5, None) is None


class TestStageNormalization:
    """Test T and N stage normalization functions."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("T1", "T1"),
            ("T2", "T2"),
            ("T3", "T3"),
            ("T4", "T4"),
            ("T1a", "T1"),  # Substage normalized
            ("T4b", "T4"),
            ("t2", "T2"),  # Case insensitive
            (None, None),
            ("TX", None),  # Unknown
            ("", None),
        ],
    )
    def test_normalize_t_stage(self, input_val, expected):
        result = normalize_t_stage(input_val)
        assert result == expected

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("N0", "N0"),
            ("N1", "N1"),
            ("N2", "N2"),
            ("N3", "N3"),
            ("N3a", "N3"),  # Substage normalized
            ("n1", "N1"),  # Case insensitive
            (None, None),
            ("NX", None),
        ],
    )
    def test_normalize_n_stage(self, input_val, expected):
        result = normalize_n_stage(input_val)
        assert result == expected


class TestStageEstimation:
    """Test tumor size and LN ratio estimation from stage."""

    def test_estimate_tumor_size_t1(self):
        """T1 tumors are typically small."""
        size = estimate_tumor_size("T1")
        assert 0 < size < 3

    def test_estimate_tumor_size_t4(self):
        """T4 tumors are typically larger."""
        size = estimate_tumor_size("T4")
        assert size > 5

    def test_estimate_ln_ratio_n0(self):
        """N0 means no or very few positive nodes."""
        ratio = estimate_ln_ratio("N0")
        assert ratio <= 0.05  # Very low ratio for N0

    def test_estimate_ln_ratio_n3(self):
        """N3 has high positive node ratio."""
        ratio = estimate_ln_ratio("N3")
        assert ratio > 0.3


class TestParseEventStatus:
    """Test disease-free status parsing."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("0:DiseaseFree", 0.0),
            ("1:Recurred/Progressed", 1.0),
            ("Tumor Free", 0.0),
            ("With Tumor", 1.0),
            ("censored", 0.0),
            ("progression", 1.0),
            (None, None),
            ("", None),
        ],
    )
    def test_parse_event_status(self, input_val, expected):
        result = parse_event_status(input_val)
        assert result == expected


class TestModelConfiguration:
    """Test model configuration loading."""

    def test_load_model_config_default(self):
        """Test loading default model config."""
        config = load_model_config()
        assert "id" in config
        assert "t_stage_weights" in config
        assert "n_stage_weights" in config
        assert "intercept" in config

    def test_load_model_config_missing_file(self):
        """Test fallback when config file is missing."""
        config = load_model_config(Path("/nonexistent/path.json"))
        # Should fall back to default
        assert config is not None
        assert "id" in config


class TestGastricCancerRiskModel:
    """Test the main risk model class."""

    @pytest.fixture
    def model(self):
        """Create model instance with default config."""
        config = load_model_config()
        return GastricCancerRiskModel(config)

    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model.t_stage_weights is not None
        assert model.n_stage_weights is not None

    def test_calculate_risk_basic(self, model):
        """Test basic risk calculation."""
        patient = {"T_stage": "T2", "N_stage": "N1", "age": 60}
        risk = model.calculate_risk(patient)
        assert 0.0 <= risk <= 1.0

    def test_calculate_risk_early_stage(self, model):
        """Test that early stage has lower risk."""
        early = {"T_stage": "T1", "N_stage": "N0", "age": 50}
        late = {"T_stage": "T4", "N_stage": "N3", "age": 70}
        risk_early = model.calculate_risk(early)
        risk_late = model.calculate_risk(late)
        assert risk_early < risk_late

    def test_calculate_risk_with_all_factors(self, model):
        """Test risk calculation with all clinical factors."""
        patient = {
            "T_stage": "T3",
            "N_stage": "N2",
            "age": 65,
            "tumor_size_cm": 5.0,
            "positive_LN": 5,
            "total_LN": 25,
        }
        risk = model.calculate_risk(patient)
        assert 0.0 <= risk <= 1.0

    def test_calculate_risk_invalid_t_stage(self, model):
        """Test that invalid T stage raises ValueError."""
        patient = {"T_stage": "T5", "N_stage": "N1"}
        with pytest.raises(ValueError, match="Unsupported T stage"):
            model.calculate_risk(patient)

    def test_calculate_risk_invalid_n_stage(self, model):
        """Test that invalid N stage raises ValueError."""
        patient = {"T_stage": "T2", "N_stage": "N5"}
        with pytest.raises(ValueError, match="Unsupported N stage"):
            model.calculate_risk(patient)

    def test_risk_category_low(self, model):
        """Test low risk category."""
        assert model.risk_category(0.15) == "Low Risk"

    def test_risk_category_moderate(self, model):
        """Test moderate risk category."""
        assert model.risk_category(0.30) == "Moderate Risk"

    def test_risk_category_high(self, model):
        """Test high risk category."""
        assert model.risk_category(0.50) == "High Risk"

    def test_risk_category_very_high(self, model):
        """Test very high risk category."""
        assert model.risk_category(0.70) == "Very High Risk"

    def test_risk_bounded_by_floor_ceiling(self, model):
        """Test that risk is bounded by floor and ceiling."""
        # Very low risk patient
        low_patient = {"T_stage": "T1", "N_stage": "N0", "age": 30}
        risk_low = model.calculate_risk(low_patient)
        assert risk_low >= model.risk_floor

        # Very high risk patient
        high_patient = {"T_stage": "T4", "N_stage": "N3", "age": 80, "ln_ratio": 0.9}
        risk_high = model.calculate_risk(high_patient)
        assert risk_high <= model.risk_ceiling


class TestScorePatients:
    """Test patient scoring functionality."""

    @pytest.fixture
    def model(self):
        config = load_model_config()
        return GastricCancerRiskModel(config)

    def test_score_single_patient(self, model):
        """Test scoring a single patient."""
        patients = [{"T_stage": "T2", "N_stage": "N1", "age": 60, "name": "Test Patient"}]
        results = score_patients(model, patients)
        
        assert len(results) == 1
        assert "Risk" in results.columns
        assert "Category" in results.columns
        assert results.iloc[0]["Patient"] == "Test Patient"

    def test_score_multiple_patients(self, model):
        """Test scoring multiple patients."""
        patients = [
            {"T_stage": "T1", "N_stage": "N0", "name": "Early"},
            {"T_stage": "T3", "N_stage": "N2", "name": "Advanced"},
        ]
        results = score_patients(model, patients)
        
        assert len(results) == 2
        assert results.iloc[0]["Risk"] < results.iloc[1]["Risk"]


class TestTCGACohort:
    """Test TCGA cohort loading and processing."""

    def test_tcga_cohort_loads_successfully(self):
        """Test that TCGA cohort can be loaded."""
        data_path = Path("data/tcga_2018_clinical_data.tsv")
        if not data_path.exists():
            pytest.skip("TCGA data file not available")
        
        cohort = load_tcga_cohort(data_path)
        assert not cohort.empty
        assert "T_stage" in cohort.columns
        assert "N_stage" in cohort.columns
        assert "age" in cohort.columns
        assert len(cohort) > 400  # TCGA STAD has ~436 patients

    def test_tcga_cohort_has_required_columns(self):
        """Test that loaded cohort has all required columns."""
        data_path = Path("data/tcga_2018_clinical_data.tsv")
        if not data_path.exists():
            pytest.skip("TCGA data file not available")
        
        cohort = load_tcga_cohort(data_path)
        required = ["T_stage", "N_stage", "age", "tumor_size_cm", "ln_ratio"]
        for col in required:
            assert col in cohort.columns, f"Missing column: {col}"

    def test_tcga_cohort_missing_file(self):
        """Test handling of missing data file."""
        cohort = load_tcga_cohort(Path("/nonexistent/path.tsv"))
        assert cohort.empty


class TestSurvivalModel:
    """Test survival model loading and predictions."""

    def test_load_survival_model(self):
        """Test loading the Han 2012 survival model."""
        model = load_survival_model()
        if model is None:
            pytest.skip("Survival model components not available")
        
        assert model.name is not None
        assert model.baseline_5yr is not None

    def test_load_survival_model_missing_file(self):
        """Test handling of missing model file."""
        model = load_survival_model(Path("/nonexistent/model.json"))
        assert model is None


class TestDualModelScoring:
    """Test combined recurrence and survival scoring."""

    @pytest.fixture
    def models(self):
        """Load both models."""
        config = load_model_config()
        recurrence = GastricCancerRiskModel(config)
        survival = load_survival_model()
        return recurrence, survival

    def test_dual_model_scoring(self, models):
        """Test scoring with both models."""
        recurrence_model, survival_model = models
        patients = [
            {"T_stage": "T2", "N_stage": "N1", "age": 60, "Sex": "Male"},
            {"T_stage": "T3", "N_stage": "N2", "age": 65, "Sex": "Female"},
        ]
        results = predict_with_both_models(patients, recurrence_model, survival_model)

        assert len(results) == 2
        assert "Risk" in results.columns
        assert all(0 <= r <= 1 for r in results["Risk"])
        
        # Check survival columns if model available
        if survival_model is not None:
            assert "survival_5yr" in results.columns
            assert "survival_10yr" in results.columns

    def test_dual_model_without_survival(self, models):
        """Test scoring without survival model."""
        recurrence_model, _ = models
        patients = [{"T_stage": "T2", "N_stage": "N1", "age": 60}]
        results = predict_with_both_models(patients, recurrence_model, survival_model=None)

        assert len(results) == 1
        assert "Risk" in results.columns
        assert "survival_5yr" not in results.columns

    def test_dual_model_risk_survival_correlation(self, models):
        """Test that recurrence risk and survival are inversely related."""
        recurrence_model, survival_model = models
        if survival_model is None:
            pytest.skip("Survival model not available")

        patients = [
            {"T_stage": "T1", "N_stage": "N0", "age": 50, "Sex": "Female"},  # Low risk
            {"T_stage": "T4", "N_stage": "N3", "age": 75, "Sex": "Male"},  # High risk
        ]
        results = predict_with_both_models(patients, recurrence_model, survival_model)

        # Higher recurrence risk should correlate with lower survival
        low_risk_row = results.iloc[0]
        high_risk_row = results.iloc[1]

        assert low_risk_row["Risk"] < high_risk_row["Risk"]
        if "survival_5yr" in results.columns:
            assert low_risk_row["survival_5yr"] > high_risk_row["survival_5yr"]


class TestEndToEndPipeline:
    """Full pipeline integration tests."""

    def test_complete_pipeline_example_patients(self):
        """Test running the example patients through the full pipeline."""
        config = load_model_config()
        model = GastricCancerRiskModel(config)
        survival_model = load_survival_model()

        patients = [
            {
                "name": "Early Stage",
                "T_stage": "T1",
                "N_stage": "N0",
                "age": 55,
                "Sex": "Female",
                "tumor_size_cm": 2.0,
                "positive_LN": 0,
                "total_LN": 20,
            },
            {
                "name": "Advanced Stage",
                "T_stage": "T3",
                "N_stage": "N2",
                "age": 68,
                "Sex": "Male",
                "tumor_size_cm": 5.0,
                "positive_LN": 8,
                "total_LN": 30,
            },
        ]

        results = predict_with_both_models(patients, model, survival_model)

        assert len(results) == 2
        assert results.iloc[0]["Category"] != results.iloc[1]["Category"]
        # Early stage should be lower risk
        assert results.iloc[0]["Risk"] < results.iloc[1]["Risk"]

    def test_tcga_cohort_pipeline(self):
        """Test processing the full TCGA cohort."""
        data_path = Path("data/tcga_2018_clinical_data.tsv")
        if not data_path.exists():
            pytest.skip("TCGA data file not available")

        # Load data
        cohort = load_tcga_cohort(data_path)
        assert len(cohort) > 0

        # Load models
        config = load_model_config()
        model = GastricCancerRiskModel(config)

        # Score a subset for speed
        sample_patients = []
        for _, row in cohort.head(10).iterrows():
            sample_patients.append(
                {
                    "name": row.get("patient_id", "Unknown"),
                    "T_stage": row["T_stage"],
                    "N_stage": row["N_stage"],
                    "age": row["age"],
                    "tumor_size_cm": row.get("tumor_size_cm"),
                    "ln_ratio": row.get("ln_ratio"),
                }
            )

        results = score_patients(model, sample_patients)
        assert len(results) == 10
        assert all(results["Risk"].notna())
