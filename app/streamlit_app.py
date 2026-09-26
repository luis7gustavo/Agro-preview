from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.getenv("AGRI_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _get(path: str, **params: Any) -> Any:
    response = httpx.get(f"{API_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.set_page_config(page_title="Motor de Decisao Agricola", layout="wide")
    st.title("Motor de Aptidao, Rentabilidade e Risco Agricola")
    st.caption(
        "Apoio a decisao com fontes publicas rastreaveis; nao substitui assistencia tecnica."
    )

    query = st.text_input("Municipio", value="Sorriso")
    try:
        options = _get("/v1/locations/search", q=query) if len(query.strip()) >= 2 else []
    except httpx.HTTPError as exc:
        st.error(f"API indisponivel em {API_URL}: {exc}")
        return
    labels = {
        f"{item['municipio']} / {item['uf']} ({item['codigo_ibge']})": item for item in options
    }
    selected_label = st.selectbox("Resultado municipal", list(labels)) if labels else None
    col1, col2, col3 = st.columns(3)
    with col1:
        area = st.number_input("Area (ha)", min_value=0.01, value=100.0)
    with col2:
        system = st.selectbox("Sistema", ["rainfed"])
    with col3:
        profile = st.selectbox("Perfil", ["balanced", "max_return", "low_risk", "fast_return"])

    if st.button("Analisar", type="primary", disabled=selected_label is None):
        selected = labels[str(selected_label)]
        try:
            response = httpx.post(
                f"{API_URL}/v1/recommendations",
                json={
                    "codigo_ibge": str(selected["codigo_ibge"]),
                    "area_ha": area,
                    "production_system": system,
                    "profile": profile,
                },
                timeout=60,
            )
            response.raise_for_status()
            st.session_state["analysis"] = response.json()
        except httpx.HTTPStatusError as exc:
            st.error(f"Analise indisponivel: {exc.response.text}")

    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("Selecione um municipio e execute a analise.")
        return
    if not analysis["recommendations"]:
        excluded = analysis.get("excluded_recommendations", [])
        st.error("A cultura foi excluida antes do ranking economico.")
        if excluded:
            st.json(excluded[0])
        with st.expander("Auditoria"):
            st.json(analysis["audit"])
        return
    recommendation = analysis["recommendations"][0]
    economics = recommendation["economics"]
    risk = recommendation["risk"]
    st.subheader(f"Soja em {analysis['location']['municipio']} / {analysis['location']['uf']}")
    temporal = recommendation.get("temporal_context", {})
    if temporal.get("mismatches"):
        st.warning(temporal["note"])
    if temporal:
        st.caption(
            f"Produtividade: {temporal['yield_reference_period']} | "
            f"Clima historico: {temporal['climate_reference_season']} | "
            f"Custo: {temporal['cost_reference_period']} | "
            f"Preco: {temporal['price_reference_period']} | "
            f"ZARC: {temporal['zarc_season']}"
        )
    climate = recommendation["climate"]
    if climate.get("origin") == "reanalysis" and climate["status"] == "available":
        st.info("Clima estimado pelo ERA5-Land para preencher uma lacuna de cobertura INMET.")
    cards = st.columns(4)
    cards[0].metric("Score", f"{recommendation['score']:.1f}")
    cards[1].metric("Lucro base / ha", f"R$ {economics['economic_profit_brl_ha']:,.2f}")
    cards[2].metric("ROI", f"{economics['roi']:.1%}")
    cards[3].metric("Prob. de lucro", f"{risk['probability_profit']:.1%}")

    tab_summary, tab_why, tab_audit = st.tabs(["Resumo", "Por que?", "Auditoria"])
    with tab_summary:
        st.json(
            {
                "produtividade": recommendation["yield"],
                "clima": recommendation["climate"],
                "periodos_utilizados": temporal,
                "preco": recommendation["price"],
                "custo": recommendation["cost"],
                "economia": economics,
                "risco": risk,
                "confianca": {
                    "score": recommendation["confidence_score"],
                    "label": recommendation["confidence_label"],
                },
            }
        )
    with tab_why:
        for reason in recommendation["reasons"]:
            st.write(f"- {reason}")
        for alert in recommendation["alerts"]:
            st.warning(alert)
    with tab_audit:
        st.json(analysis["audit"])


if __name__ == "__main__":
    main()
