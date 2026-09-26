from agri_decision.transforms.conab import parse_soy_costs, parse_soy_prices


def test_cost_mapping_preserves_components_and_builds_documented_totals() -> None:
    content = (
        "empreendimento;ano;mes;ano_mes;produto;id_produto;safra;uf;municipio;cod_ibge;"
        "unidade_comercializacao;vlr_custo_variavel_ha;vlr_custo_variavel_unidade;"
        "vlr_custo_fixo_ha;vlr_custo_fixo_unidade;vlr_renda_fator_ha;"
        "vlr_renda_fator_unidade\n"
        "SEQUEIRO;2026;3;202603;SOJA;1;2025/26;MT;Sorriso;5107925;60 KG;"
        "4000.00;60.00;800.00;12.00;200.00;3.00\n"
    ).encode("cp1252")
    row = parse_soy_costs(content, source_checksum="a" * 64).to_dicts()[0]
    assert row["custo_operacional_brl_ha"] == 4_800
    assert row["custo_total_brl_ha"] == 5_000
    assert row["data_nature"] == "observed"


def test_price_filter_requires_exact_soy_product() -> None:
    header = (
        "produto;classificao_produto;id_produto;nom_municipio;cod_ibge;uf;regiao;ano;mes;"
        "dsc_nivel_comercializacao;valor_produto_kg\n"
    )
    rows = (
        "SOJA;;1;Sorriso;5107925;MT;CO;2026;7;ATACADO;2,10\n"
        "FARELO DE SOJA;;2;Sorriso;5107925;MT;CO;2026;7;ATACADO;1,50\n"
    )
    frame = parse_soy_prices(
        (header + rows).encode("cp1252"),
        coverage_level="municipality",
        source_checksum="b" * 64,
    )
    assert frame.height == 1
    assert frame.item(0, "preco_brl_kg") == 2.10
