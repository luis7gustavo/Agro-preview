import pytest

from agri_decision.provenance.artifacts import sha256_bytes
from agri_decision.taxonomy import CropTaxonomy
from agri_decision.transforms.ibge_pam import (
    _municipality_and_uf,
    parse_pam_rows,
    parse_sidra_number,
)


def sidra_rows() -> list[dict[str, str]]:
    # Synthetic SIDRA-shaped fixture; never used outside tests.
    header = {
        "NC": "Nível Territorial (Código)",
        "NN": "Nível Territorial",
        "MC": "Unidade de Medida (Código)",
        "MN": "Unidade de Medida",
        "V": "Valor",
        "D1C": "Município (Código)",
        "D1N": "Município",
        "D2C": "Variável (Código)",
        "D2N": "Variável",
        "D3C": "Ano (Código)",
        "D3N": "Ano",
        "D4C": "Produto (Código)",
        "D4N": "Produto",
    }
    variables = [
        ("8331", "Área plantada", "Hectares", "100"),
        ("216", "Área colhida", "Hectares", "90"),
        ("214", "Quantidade produzida", "Toneladas", "270"),
        ("112", "Rendimento médio", "Quilogramas por Hectare", "3000"),
        ("215", "Valor da produção", "Mil Reais", "450"),
    ]
    rows = [header]
    for variable, name, unit, value in variables:
        rows.append(
            {
                "NC": "6",
                "NN": "Município",
                "MC": "test",
                "MN": unit,
                "V": value,
                "D1C": "1234567",
                "D1N": "Município de Teste (DF)",
                "D2C": variable,
                "D2N": name,
                "D3C": "2024",
                "D3N": "2024",
                "D4C": "40124",
                "D4N": "Soja (em grão)",
            }
        )
    return rows


def test_parser_normalizes_units_and_provenance() -> None:
    raw = sidra_rows()
    checksum = sha256_bytes(b"synthetic-test-fixture")

    records = parse_pam_rows(raw, taxonomy=CropTaxonomy.from_file(), source_checksum=checksum)

    assert len(records) == 1
    record = records[0]
    assert record.produtividade_kg_ha == 3000
    assert record.valor_producao_original == 450
    assert record.valor_producao_unidade_original == "Mil Reais"
    assert record.valor_producao_brl == 450_000
    assert record.source_dataset_checksum == checksum
    assert record.quality_flags == []


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Sorriso (MT)", ("Sorriso", "MT")),
        ("Alta Floresta D'Oeste - RO", ("Alta Floresta D'Oeste", "RO")),
    ],
)
def test_parser_accepts_both_official_sidra_municipality_labels(
    label: str,
    expected: tuple[str, str],
) -> None:
    assert _municipality_and_uf(label) == expected


def test_historical_currency_is_preserved_but_not_labeled_as_brl() -> None:
    raw = sidra_rows()
    for row in raw[1:]:
        row["D3C"] = "1974"
        row["D3N"] = "1974"
        if row["D2C"] == "8331":
            row["MN"] = ""
            row["V"] = "..."
        if row["D2C"] == "215":
            row["MN"] = "Mil Cruzeiros"
    checksum = sha256_bytes(b"synthetic-historical-currency-fixture")

    record = parse_pam_rows(
        raw,
        taxonomy=CropTaxonomy.from_file(),
        source_checksum=checksum,
    )[0]

    assert record.valor_producao_original == 450
    assert record.valor_producao_unidade_original == "Mil Cruzeiros"
    assert record.valor_producao_brl is None
    assert "valor_producao_brl:HISTORICAL_CURRENCY_NOT_NORMALIZED" in record.quality_flags


@pytest.mark.parametrize(
    ("token", "value", "flag"),
    [
        ("-", 0.0, "SIDRA_ABSOLUTE_ZERO"),
        ("0", 0.0, "SIDRA_ROUNDED_ZERO"),
        ("X", None, "SIDRA_SUPPRESSED"),
        ("..", None, "SIDRA_NOT_APPLICABLE"),
        ("...", None, "SIDRA_NOT_AVAILABLE"),
    ],
)
def test_special_sidra_values_are_not_silently_coerced(
    token: str,
    value: float | None,
    flag: str,
) -> None:
    parsed = parse_sidra_number(token)

    assert parsed.value == value
    assert parsed.flag == flag
