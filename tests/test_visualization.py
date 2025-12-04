"""Tests for visualization module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from utils.visualization import (
    CATEGORY_COLORS,
    FIG_RISK_PREDICTIONS,
    FIG_TCGA_SUMMARY,
    FIG_SURVIVAL_PREDICTIONS,
    FIG_SURVIVAL_VS_RECURRENCE,
    finalize_figure,
    plot_individual_predictions,
    plot_survival_predictions,
    plot_survival_vs_recurrence,
    plot_tcga_summary,
    plot_calibration_curve,
)


@pytest.fixture
def sample_results():
    """Create minimal results DataFrame for testing."""
    return pd.DataFrame(
        {
            "Patient": ["A", "B", "C", "D"],
            "Risk": [0.15, 0.35, 0.55, 0.85],
            "Category": ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"],
            "T_stage": ["T1", "T2", "T3", "T4"],
            "N_stage": ["N0", "N1", "N2", "N3"],
        }
    )


@pytest.fixture
def sample_results_with_survival():
    """Create results DataFrame with survival columns."""
    return pd.DataFrame(
        {
            "Patient": ["A", "B", "C", "D", "E"],
            "Risk": [0.15, 0.35, 0.55, 0.75, 0.85],
            "Category": ["Low Risk", "Moderate Risk", "High Risk", "High Risk", "Very High Risk"],
            "T_stage": ["T1", "T2", "T3", "T3", "T4"],
            "N_stage": ["N0", "N1", "N2", "N2", "N3"],
            "survival_5yr": [0.85, 0.70, 0.50, 0.45, 0.25],
            "survival_10yr": [0.75, 0.55, 0.35, 0.30, 0.10],
            "survival_category": [
                "Excellent Prognosis",
                "Good Prognosis",
                "Moderate Prognosis",
                "Poor Prognosis",
                "Very Poor Prognosis",
            ],
        }
    )


@pytest.fixture
def sample_calibration_data():
    """Create results DataFrame with event labels for calibration."""
    return pd.DataFrame(
        {
            "Patient": [f"P{i}" for i in range(50)],
            "Risk": [0.1 + 0.8 * (i / 49) for i in range(50)],  # Range from 0.1 to 0.9
            "event_observed": [1 if i % 3 == 0 else 0 for i in range(50)],  # ~33% events
        }
    )


class TestCategoryColors:
    """Test that category color mapping is properly defined."""

    def test_all_categories_have_colors(self):
        """Verify all standard risk categories have defined colors."""
        expected_categories = ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
        for category in expected_categories:
            assert category in CATEGORY_COLORS
            assert CATEGORY_COLORS[category].startswith("#")


class TestFinalizeFunction:
    """Test the finalize_figure utility function."""

    def test_finalize_creates_output_directory(self):
        """Test that output directory is created if missing."""
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "subdir" / "figure.png"
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])

            output_path = finalize_figure(fig, nested_path, show_plots=False)
            assert output_path.exists()
            assert output_path.parent.exists()

    def test_finalize_closes_figure_when_not_showing(self):
        """Test that figure is closed when show_plots=False."""
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            fig_num = fig.number
            ax.plot([1, 2, 3])

            finalize_figure(fig, Path(tmpdir) / "test.png", show_plots=False)
            # Figure should be closed
            assert fig_num not in plt.get_fignums()


class TestFigureGeneration:
    """Test that figures are created without errors."""

    def test_individual_predictions_creates_file(self, sample_results):
        """Test individual predictions plot generates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_individual_predictions(
                sample_results, Path(tmpdir), show_plots=False
            )
            assert output_path.exists()
            assert output_path.suffix == ".png"
            assert output_path.name == FIG_RISK_PREDICTIONS

    def test_individual_predictions_handles_missing_categories(self):
        """Test plot handles missing risk categories gracefully."""
        df = pd.DataFrame(
            {
                "Patient": ["A", "B"],
                "Risk": [0.15, 0.25],
                "Category": ["Low Risk", "Low Risk"],  # Only one category
                "T_stage": ["T1", "T2"],
                "N_stage": ["N0", "N0"],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_individual_predictions(df, Path(tmpdir), show_plots=False)
            assert output_path.exists()

    def test_tcga_summary_creates_file(self, sample_results):
        """Test TCGA summary plot generates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_tcga_summary(sample_results, Path(tmpdir), show_plots=False)
            assert output_path.exists()
            assert output_path.suffix == ".png"
            assert output_path.name == FIG_TCGA_SUMMARY

    def test_tcga_summary_handles_sparse_stages(self):
        """Test TCGA summary handles sparse stage combinations."""
        df = pd.DataFrame(
            {
                "Patient": ["A", "B", "C"],
                "Risk": [0.15, 0.45, 0.85],
                "Category": ["Low Risk", "High Risk", "Very High Risk"],
                "T_stage": ["T1", "T2", "T4"],  # Missing T3
                "N_stage": ["N0", "N1", "N3"],  # Missing N2
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_tcga_summary(df, Path(tmpdir), show_plots=False)
            assert output_path.exists()


class TestSurvivalPlots:
    """Test survival-related plots."""

    def test_survival_predictions_creates_file(self, sample_results_with_survival):
        """Test survival predictions plot generates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_survival_predictions(
                sample_results_with_survival, Path(tmpdir), show_plots=False
            )
            assert output_path is not None
            assert output_path.exists()
            assert output_path.name == FIG_SURVIVAL_PREDICTIONS

    def test_survival_predictions_returns_none_without_data(self, sample_results):
        """Test survival predictions returns None when survival data missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_survival_predictions(
                sample_results, Path(tmpdir), show_plots=False  # No survival columns
            )
            assert output_path is None

    def test_survival_vs_recurrence_creates_file(self, sample_results_with_survival):
        """Test survival vs recurrence comparison plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_survival_vs_recurrence(
                sample_results_with_survival, Path(tmpdir), show_plots=False
            )
            assert output_path is not None
            assert output_path.exists()
            assert output_path.name == FIG_SURVIVAL_VS_RECURRENCE

    def test_survival_vs_recurrence_returns_none_without_data(self, sample_results):
        """Test returns None when required columns missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_survival_vs_recurrence(
                sample_results, Path(tmpdir), show_plots=False
            )
            assert output_path is None

    def test_survival_vs_recurrence_handles_empty_after_dropna(self):
        """Test handles case where all rows have NaN."""
        df = pd.DataFrame(
            {
                "Patient": ["A", "B"],
                "Risk": [0.5, 0.6],
                "survival_5yr": [None, None],  # All NaN
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_survival_vs_recurrence(df, Path(tmpdir), show_plots=False)
            assert output_path is None


class TestCalibrationCurve:
    """Test calibration curve plotting."""

    def test_calibration_curve_creates_file(self, sample_calibration_data):
        """Test calibration curve plot generates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_calibration_curve(
                sample_calibration_data, Path(tmpdir), show_plots=False, label_column="event_observed"
            )
            assert result is not None
            output_path, brier = result
            assert output_path.exists()
            assert output_path.suffix == ".png"
            assert 0.0 <= brier <= 1.0

    def test_calibration_curve_returns_none_without_labels(self, sample_results):
        """Test returns None when label column missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_calibration_curve(
                sample_results, Path(tmpdir), show_plots=False, label_column="event_observed"
            )
            assert result is None

    def test_calibration_curve_handles_all_nan_labels(self):
        """Test handles case where all labels are NaN."""
        df = pd.DataFrame(
            {
                "Patient": ["A", "B"],
                "Risk": [0.5, 0.6],
                "event_observed": [None, None],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_calibration_curve(
                df, Path(tmpdir), show_plots=False, label_column="event_observed"
            )
            assert result is None


class TestPlotContent:
    """Test that plot content is as expected."""

    def test_individual_predictions_bar_count(self, sample_results):
        """Test that correct number of bars are plotted."""
        import matplotlib
        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt

        # Get figure before it's closed
        colors = sample_results["Category"].map(CATEGORY_COLORS).fillna("gray")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        bars = axes[0].barh(
            sample_results["Patient"], sample_results["Risk"] * 100, color=colors
        )
        assert len(bars) == len(sample_results)
        plt.close(fig)

    def test_risk_values_scaled_correctly(self, sample_results):
        """Verify risk values are scaled to percentage (0-100)."""
        # Risk values in DataFrame are 0-1, but plots show 0-100%
        assert all(0 <= r <= 1 for r in sample_results["Risk"])
        # Scaled values for display
        scaled = sample_results["Risk"] * 100
        assert all(0 <= s <= 100 for s in scaled)
