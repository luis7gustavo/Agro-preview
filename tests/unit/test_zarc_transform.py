import polars as pl

from agri_decision.transforms.zarc import (
    build_zarc_municipality_coverage,
    parse_soy_zarc_windows,
)


def _fixture() -> bytes:
    decs = ";".join(f"dec{i}" for i in range(1, 37))
    values = ["20", "0", "30"] + ["0"] * 33
    header = (
        "Nome_cultura;SafraIni;SafraFin;Cod_Cultura;Cod_Ciclo;Cod_Solo;geocodigo;UF;"
        "municipio;Cod_Clima;Nome_Clima;Cod_Outros_Manejos;Nome_Outros_Manejos;"
        f"Produtividade;Cod_NM;Cod_Munic;Cod_Meso;Cod_Micro;Portaria;{decs}\n"
    )
    row = (
        "Soja;2026;2027;11;20;1;5107925;MT;Sorriso;0;NA;1;Sequeiro;;;;;;"
        f"Port.202;{';'.join(values)}\n"
    )
    return (header + row).encode()


def test_zarc_is_long_and_only_preserves_eligible_decendios() -> None:
    windows = parse_soy_zarc_windows(_fixture(), source_checksum="a" * 64)
    assert windows.get_column("decendio").to_list() == [1, 3]
    assert windows.get_column("nivel_risco").to_list() == [20, 30]


def test_coverage_distinguishes_not_eligible_from_not_available() -> None:
    windows = parse_soy_zarc_windows(_fixture(), source_checksum="a" * 64)
    locations = pl.DataFrame(
        {
            "codigo_ibge": ["5107925", "5100001", "1200001"],
            "municipio": ["Sorriso", "Outro MT", "Outro AC"],
            "uf": ["MT", "MT", "AC"],
        }
    )
    coverage = build_zarc_municipality_coverage(windows, locations)
    status = dict(zip(coverage["codigo_ibge"], coverage["zarc_status"], strict=True))
    assert status == {
        "1200001": "not_available",
        "5100001": "not_eligible",
        "5107925": "eligible",
    }
