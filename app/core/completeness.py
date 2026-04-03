"""Deterministic data completeness checker.

Checks ``snapshot_data_json`` for missing fields required by ICM 2018
scoring and clinical decision-making.  No LLM involved — purely
rule-based.
"""

from typing import Any


# ------------------------------------------------------------------
# Helper accessors
# ------------------------------------------------------------------

def _get_nested(data: dict, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts."""
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
    inflammatory = latest.get("inflammatory_markers_blood", {})
    if inflammatory.get(field_name) is not None:
        return True
    synovial = latest.get("synovial_fluid", {})
    return synovial.get(field_name) is not None


def _has_synovial_field(snap: dict, field_name: str) -> bool:
    synovial = _get_nested(snap, "lab_results", "latest", "synovial_fluid")
    if not synovial:
        return False
    return synovial.get(field_name) is not None


def _has_histology(snap: dict) -> bool:
    items = _get_nested(snap, "culture_results", "items", default=[])
    if isinstance(items, list):
        for item in items:
            sample = item.get("sample_type", "").lower()
            if any(kw in sample for kw in ("giai phau benh", "histolog", "patholog")):
                return True

    surgeries = _get_nested(snap, "surgeries", "items", default=[])
    if isinstance(surgeries, list):
        for s in surgeries:
            findings = (s.get("findings") or "").lower()
            if any(kw in findings for kw in ("giai phau benh", "sinh thiet", "histolog")):
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


# ------------------------------------------------------------------
# Check definitions
# ------------------------------------------------------------------

ICM_MAJOR_CHECKS: list[dict[str, Any]] = [
    {
        "field": "sinus_tract",
        "category": "ICM_MAJOR",
        "importance": "CRITICAL",
        "message": (
            "Duong ro thong voi khop gia (major criterion) "
            "— ket luan INFECTED ngay neu duong tinh"
        ),
        "check": _has_sinus_tract_data,
    },
    {
        "field": "culture_results",
        "category": "ICM_MAJOR",
        "importance": "CRITICAL",
        "message": (
            "Khong co ket qua nuoi cay "
            "— can it nhat 2 mau de danh gia major criterion"
        ),
        "check": _has_culture_results,
    },
]

ICM_MINOR_CHECKS: list[dict[str, Any]] = [
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
        "message": (
            "Giai phau benh mo quanh khop (3 diem ICM) "
            "— can sinh thiet mo trong phau thuat"
        ),
        "check": _has_histology,
    },
]

CLINICAL_CHECKS: list[dict[str, Any]] = [
    {
        "field": "infection_type",
        "category": "CLINICAL",
        "importance": "HIGH",
        "message": (
            "Chua xac dinh ACUTE/CHRONIC "
            "— anh huong nguong ICM va chien luoc phau thuat"
        ),
        "check": _has_infection_type,
    },
    {
        "field": "implant_stability",
        "category": "CLINICAL",
        "importance": "HIGH",
        "message": "Chua danh gia on dinh implant — anh huong chon DAIR vs revision",
        "check": _has_implant_stability,
    },
    {
        "field": "allergies",
        "category": "CLINICAL",
        "importance": "HIGH",
        "message": (
            "Chua co thong tin di ung thuoc "
            "— can thiet de chon phac do khang sinh an toan"
        ),
        "check": _has_allergy_info,
    },
    {
        "field": "renal_function",
        "category": "CLINICAL",
        "importance": "MEDIUM",
        "message": "Chua co creatinine/eGFR — can de chinh lieu Vancomycin va TMP-SMX",
        "check": _has_renal_function,
    },
    {
        "field": "liver_function",
        "category": "CLINICAL",
        "importance": "MEDIUM",
        "message": "Chua co AST/ALT — can de danh gia an toan Rifampicin",
        "check": _has_liver_function,
    },
]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def check_data_completeness(snapshot_data: dict[str, Any]) -> dict[str, Any]:
    """Check *snapshot_data* for missing fields required by ICM 2018.

    Returns:
        Dict with ``is_complete``, ``missing_items``,
        ``completeness_score``, and ``impact_note``.
    """
    missing_items: list[dict[str, str]] = []
    all_checks = ICM_MAJOR_CHECKS + ICM_MINOR_CHECKS + CLINICAL_CHECKS

    for check_def in all_checks:
        if not check_def["check"](snapshot_data):
            missing_items.append({
                "field": check_def["field"],
                "category": check_def["category"],
                "importance": check_def["importance"],
                "message": check_def["message"],
            })

    icm_minor_total = len(ICM_MINOR_CHECKS)
    icm_minor_present = icm_minor_total - sum(
        1 for m in missing_items if m["category"] == "ICM_MINOR"
    )

    is_complete = len(missing_items) == 0
    critical_missing = sum(1 for m in missing_items if m["importance"] == "CRITICAL")

    if critical_missing > 0:
        impact_note = (
            f"Thieu {critical_missing} du lieu CRITICAL — ket qua chan doan "
            f"va phac do co the khong chinh xac. "
            f"Khuyen nghi bo sung truoc khi ap dung."
        )
    elif missing_items:
        impact_note = (
            f"Thieu {len(missing_items)} du lieu — ket qua van hop le "
            f"nhung do chinh xac se tang neu bo sung them."
        )
    else:
        impact_note = (
            "Du lieu day du — ket qua chan doan va phac do co do tin cay cao."
        )

    return {
        "is_complete": is_complete,
        "missing_items": missing_items,
        "completeness_score": (
            f"{icm_minor_present}/{icm_minor_total} "
            f"ICM minor criteria co du lieu"
        ),
        "impact_note": impact_note,
    }
