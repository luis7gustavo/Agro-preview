from pathlib import Path

from agri_decision.dimensions import build_dim_crop
from agri_decision.schemas.crop import ImplementationStatus
from agri_decision.taxonomy import CropTaxonomy, normalize_crop_label

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def taxonomy() -> CropTaxonomy:
    return CropTaxonomy.from_file(PROJECT_ROOT / "configs" / "crops.yml")


def test_normalization_is_accent_and_punctuation_insensitive() -> None:
    assert normalize_crop_label("  Soja (em GRÃO) ") == "soja em grao"


def test_source_alias_resolves_to_canonical_crop() -> None:
    assert taxonomy().resolve("Soja (em grão)", source="ibge") == "soja"
    assert taxonomy().resolve("ALGODÃO", source="conab") == "algodao"


def test_six_crops_are_configured_but_only_soy_is_active_pilot() -> None:
    crops = build_dim_crop(taxonomy())
    assert len(crops) == 6
    active = [crop for crop in crops if crop.active]
    assert [crop.canonical_name for crop in active] == ["soja"]
    assert active[0].implementation_status == ImplementationStatus.PILOT
