"""Tests for the deterministic data-completeness checker."""

from app.core.completeness import check_data_completeness


class TestCheckDataCompleteness:
    """Unit tests for ``check_data_completeness``."""

    def test_empty_snapshot_reports_all_missing(self, empty_snapshot: dict):
        result = check_data_completeness(empty_snapshot)
        assert result["is_complete"] is False
        assert len(result["missing_items"]) > 0

    def test_complete_snapshot_has_few_missing(self, sample_snapshot: dict):
        result = check_data_completeness(sample_snapshot)
        missing_fields = [m["field"] for m in result["missing_items"]]
        # The sample snapshot has most fields present.
        # Only histology and leukocyte_esterase are missing.
        assert "serum_CRP" not in missing_fields
        assert "serum_ESR" not in missing_fields
        assert "synovial_WBC" not in missing_fields
        assert "culture_results" not in missing_fields
        assert "allergies" not in missing_fields

    def test_completeness_score_format(self, sample_snapshot: dict):
        result = check_data_completeness(sample_snapshot)
        assert "ICM minor criteria" in result["completeness_score"]

    def test_missing_culture_results_detected(self):
        snap = {
            "clinical_records": {"symptoms": {"sinus_tract": False}},
            "culture_results": {"items": [{"name": "S. aureus"}]},
        }
        result = check_data_completeness(snap)
        missing_fields = [m["field"] for m in result["missing_items"]]
        assert "culture_results" in missing_fields  # needs >= 2 items

    def test_critical_missing_impact_note(self, empty_snapshot: dict):
        result = check_data_completeness(empty_snapshot)
        assert "CRITICAL" in result["impact_note"]

    def test_full_completeness(self, sample_snapshot: dict):
        """When leukocyte_esterase and histology are added, fewer missing."""
        snap = sample_snapshot.copy()
        snap["lab_results"]["latest"]["synovial_fluid"]["leukocyte_esterase"] = "++"
        snap["culture_results"]["items"].append(
            {"sample_type": "Giai phau benh", "result_status": "POSITIVE"},
        )
        result = check_data_completeness(snap)
        missing_fields = [m["field"] for m in result["missing_items"]]
        assert "synovial_LE" not in missing_fields
        assert "positive_histology" not in missing_fields
