"""Deterministic Data Completeness Checker.

Checks snapshot_data_json for missing fields required by ICM 2018 scoring
and clinical decision-making. No LLM involved - purely rule-based.
"""

from typing import Any


# ICM 2018 major criteria fields
ICM_MAJOR_CHECKS = [
    {
        "field": "sinus_tract",
        "category": "ICM_MAJOR",
        "importance": "CRITICAL",
        "message": "Duong ro thong voi khop gia (major criterion) — ket luan INFECTED ngay neu duong tinh",
        "check": lambda snap: _has_sinus_tract_data(snap),
    },
    {
        "field": "culture_results",
        "category": "ICM_MAJOR",
        "importance": "CRITICAL",
        "message": "Khong co ket qua nuoi cay — can it nhat 2 mau de danh gia major criterion",
        "check": lambda snap: _has_culture_results(snap),
    },
]

# ICM 2018 minor criteria fields
ICM_MINOR_CHECKS = [
    {
        "field": "serum_CRP",
        "category": "ICM_MINOR",
        "importance": "HIGH",
        "message": "CRP huyet thanh (2 diem ICM) — marker viem he thong",
        "check": lambda snap: _has_lab_field(snap, "crp"),
    },
    {
        "field": "serum_ESR",
        "category": "ICM_MINOR",
        "importance": "HIGH",
        "message": "Toc do mau lang ESR (1 diem ICM)",
        "check": lambda snap: _has_lab_field(snap, "esr"),
    },
    {
        "field": "serum_D_Dimer",
        "category": "ICM_MINOR",
        "importance": "HIGH",
        "message": "D-Dimer (2 diem ICM, ket hop CRP)",
        "check": lambda snap: _has_lab_field(snap, "d_dimer"),
    },
    {
        "field": "serum_IL6",
        "category": "ICM_MINOR",
        "importance": "MEDIUM",
        "message": "IL-6 huyet thanh (1 diem ICM)",
        "check": lambda snap: _has_lab_field(snap, "serum_il6"),
    },
    {
        "field": "synovial_WBC",
        "category": "ICM_MINOR",
        "importance": "CRITICAL",
        "message": "Bach cau dich khop (3 diem ICM) — marker quan trong nhat",
        "check": lambda snap: _has_synovial_field(snap, "synovial_wbc"),
    },
    {
        "field": "synovial_PMN",
        "category": "ICM_MINOR",
        "importance": "HIGH",
        "message": "PMN% dich khop (2 diem ICM)",
        "check": lambda snap: _has_synovial_field(snap, "synovial_pmn"),
    },
    {
        "field": "synovial_alpha_defensin",
        "category": "ICM_MINOR",
        "importance": "HIGH",
        "message": "Alpha-Defensin (3 diem ICM) — do dac hieu >96%",
        "check": lambda snap: _has_lab_field(snap, "alpha_defensin"),
    },
    {
        "field": "synovial_LE",
        "category": "ICM_MINOR",
        "importance": "MEDIUM",
        "message": "Leukocyte Esterase dich khop (3 diem ICM) — test nhanh tai cho",
        "check": lambda snap: _has_lab_field(snap, "leukocyte_esterase"),
    },
    {
        "field": "positive_histology",
        "category": "ICM_MINOR",
        "importance": "HIGH",
        "message": "Giai phau benh mo quanh khop (3 diem ICM) — can sinh thiet mo trong phau thuat",
        "check": lambda snap: _has_histology(snap),
    },
]

# Clinical fields needed for treatment strategy
CLINICAL_CHECKS = [
    {
        "field": "infection_type",
        "category": "CLINICAL",
        "importance": "HIGH",
        "message": "Chua xac dinh ACUTE/CHRONIC — anh huong nguong ICM va chien luoc phau thuat",
        "check": lambda snap: _has_infection_type(snap),
    },
    {
        "field": "implant_stability",
        "category": "CLINICAL",
        "importance": "HIGH",
        "message": "Chua danh gia on dinh implant — anh huong chon DAIR vs revision",
        "check": lambda snap: _has_implant_stability(snap),
    },
    {
        "field": "allergies",
        "category": "CLINICAL",
        "importance": "HIGH",
        "message": "Chua co thong tin di ung thuoc — can thiet de chon phac do khang sinh an toan",
        "check": lambda snap: _has_allergy_info(snap),
    },
    {
        "field": "renal_function",
        "category": "CLINICAL",
        "importance": "MEDIUM",
        "message": "Chua co creatinine/eGFR — can de chinh lieu Vancomycin va TMP-SMX",
        "check": lambda snap: _has_renal_function(snap),
    },
    {
        "field": "liver_function",
        "category": "CLINICAL",
        "importance": "MEDIUM",
        "message": "Chua co AST/ALT — can de danh gia an toan Rifampicin",
        "check": lambda snap: _has_liver_function(snap),
    },
]


def check_data_completeness(snapshot_data: dict[str, Any]) -> dict[str, Any]:
    """Check snapshot data for missing fields required by ICM 2018 and clinical decisions.

    Returns:
        dict with is_complete, missing_items, completeness_score, impact_note
    """
    missing_items = []
    all_checks = ICM_MAJOR_CHECKS + ICM_MINOR_CHECKS + CLINICAL_CHECKS

    for check_def in all_checks:
        if not check_def["check"](snapshot_data):
            missing_items.append({
                "field": check_def["field"],
                "category": check_def["category"],
                "importance": check_def["importance"],
                "message": check_def["message"],
            })

    # Calculate ICM minor completeness
    icm_minor_total = len(ICM_MINOR_CHECKS)
    icm_minor_present = icm_minor_total - sum(
        1 for item in missing_items if item["category"] == "ICM_MINOR"
    )

    is_complete = len(missing_items) == 0
    critical_missing = sum(1 for m in missing_items if m["importance"] == "CRITICAL")

    impact_note = ""
    if critical_missing > 0:
        impact_note = (
            f"Thieu {critical_missing} du lieu CRITICAL — ket qua chan doan "
            f"va phac do co the khong chinh xac. Khuyen nghi bo sung truoc khi ap dung."
        )
    elif len(missing_items) > 0:
        impact_note = (
            f"Thieu {len(missing_items)} du lieu — ket qua van hop le "
            f"nhung do chinh xac se tang neu bo sung them."
        )
    else:
        impact_note = "Du lieu day du — ket qua chan doan va phac do co do tin cay cao."

    return {
        "is_complete": is_complete,
        "missing_items": missing_items,
        "completeness_score": f"{icm_minor_present}/{icm_minor_total} ICM minor criteria co du lieu",
        "impact_note": impact_note,
    }


# ==================== Helper check functions ====================

def _get_nested(data: dict, *keys, default=None):
    """Safely get nested dict value."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def _has_sinus_tract_data(snap: dict) -> bool:
    symptoms = _get_nested(snap, "clinical_records", "symptoms")
    if not symptoms:
        return False
    return "sinus_tract" in symptoms


def _has_culture_results(snap: dict) -> bool:
    items = _get_nested(snap, "culture_results", "items")
    return isinstance(items, list) and len(items) >= 2


def _has_lab_field(snap: dict, field_name: str) -> bool:
    latest = _get_nested(snap, "lab_results", "latest")
    if not latest:
        return False
    # Check in inflammatory_markers_blood
    inflammatory = latest.get("inflammatory_markers_blood", {})
    if inflammatory.get(field_name) is not None:
        return True
    # Check in synovial_fluid
    synovial = latest.get("synovial_fluid", {})
    if synovial.get(field_name) is not None:
        return True
    return False


def _has_synovial_field(snap: dict, field_name: str) -> bool:
    synovial = _get_nested(snap, "lab_results", "latest", "synovial_fluid")
    if not synovial:
        return False
    return synovial.get(field_name) is not None


def _has_histology(snap: dict) -> bool:
    """Check if histology/pathology results exist."""
    # Check in culture items for histology
    items = _get_nested(snap, "culture_results", "items", default=[])
    if isinstance(items, list):
        for item in items:
            sample = item.get("sample_type", "").lower()
            if "giai phau benh" in sample or "histolog" in sample or "patholog" in sample:
                return True
    # Check in surgeries for biopsy findings
    surgeries = _get_nested(snap, "surgeries", "items", default=[])
    if isinstance(surgeries, list):
        for s in surgeries:
            findings = (s.get("findings") or "").lower()
            if "giai phau benh" in findings or "sinh thiet" in findings or "histolog" in findings:
                return True
    return False


def _has_infection_type(snap: dict) -> bool:
    infection = _get_nested(snap, "clinical_records", "infection_assessment")
    if not infection:
        return False
    return infection.get("suspected_infection_type") is not None


def _has_implant_stability(snap: dict) -> bool:
    infection = _get_nested(snap, "clinical_records", "infection_assessment")
    if not infection:
        return False
    return infection.get("implant_stability") is not None


def _has_allergy_info(snap: dict) -> bool:
    allergies = _get_nested(snap, "medical_history", "allergies")
    if not allergies:
        return False
    return "is_allergy" in allergies


def _has_renal_function(snap: dict) -> bool:
    biochem = _get_nested(snap, "lab_results", "latest", "biochemical_data")
    if not biochem:
        return False
    return biochem.get("creatinine") is not None


def _has_liver_function(snap: dict) -> bool:
    biochem = _get_nested(snap, "lab_results", "latest", "biochemical_data")
    if not biochem:
        return False
    return biochem.get("alt") is not None or biochem.get("ast") is not None
