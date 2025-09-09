
# streamlit_app_pro.py
import json
import math
import os
from io import BytesIO
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# --- Backends headless-safe (Matplotlib font cache) ---
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
try:
    import matplotlib
    matplotlib.use("Agg", force=True)
except Exception:
    pass

# =========================
# Config padrão (editável)
# =========================
DEFAULT_CONFIG = {
    "metas": {"mencoes": 500, "engajamentos": 6000, "sentimento": 70.0, "brandfit": 7.0},
    "pesos_iniciais": {"mencoes": 0.5, "engajamentos": 2.0, "sentimento": 1.0, "brandfit": 0.75},
    "caps": {"ratio_cap": 2.0},
    "logistic_k": 4.0,
}

# ---------- Utils de cálculo (mantidos do app original) ----------
def normalize_features(mencoes, engaj, sentimento, brandfit, CONFIG):
    metas = CONFIG["metas"]
    caps = CONFIG["caps"]
    r_menc = min(max(0.0, mencoes / max(1.0, metas["mencoes"])), caps["ratio_cap"])
    r_eng = min(max(0.0, engaj / max(1.0, metas["engajamentos"])), caps["ratio_cap"])
    r_sent = min(max(0.0, sentimento / max(1e-6, metas["sentimento"])), caps["ratio_cap"])
    r_bfit = min(max(0.0, brandfit / max(1e-6, metas["brandfit"])), caps["ratio_cap"])

    menc_log = math.log1p(mencoes) / math.log1p(metas["mencoes"])
    eng_log = math.log1p(engaj) / math.log1p(metas["engajamentos"])

    sent_smooth = math.tanh(r_sent)
    bfit_smooth = math.tanh(r_bfit)

    return np.array(
        [r_menc, menc_log, r_eng, eng_log, r_sent, sent_smooth, r_bfit, bfit_smooth],
        dtype=float,
    )


def compute_composite(features, CONFIG):
    pesos = CONFIG["pesos_iniciais"]
    r_menc, menc_log, r_eng, eng_log, r_sent, sent_smooth, r_bfit, bfit_smooth = features

    comp_menc = 0.5 * r_menc + 0.5 * menc_log
    comp_eng = 0.5 * r_eng + 0.5 * eng_log
    comp_sent = 0.5 * r_sent + 0.5 * sent_smooth
    comp_bfit = 0.5 * r_bfit + 0.5 * bfit_smooth

    w = np.array(
        [pesos["mencoes"], pesos["engajamentos"], pesos["sentimento"], pesos["brandfit"]],
        dtype=float,
    )
    w = w / (w.sum() if w.sum() != 0 else 1.0)

    composite = w[0] * comp_menc + w[1] * comp_eng + w[2] * comp_sent + w[3] * comp_bfit
    comps = {"mencoes": comp_menc, "engajamentos": comp_eng, "sentimento": comp_sent, "brandfit": comp_bfit}
    weights = {"mencoes": w[0], "engajamentos": w[1], "sentimento": w[2], "brandfit": w[3]}
    return composite, comps, weights


def composite_to_score(composite, CONFIG):
    k = CONFIG["logistic_k"]
    prob_like = 1.0 / (1.0 + math.exp(-k * (composite - 1.0)))
    return float(100.0 * prob_like)


def score_to_composite_target(score_target, CONFIG):
    k = CONFIG["logistic_k"]
    p = max(1e-6, min(1 - 1e-6, score_target / 100.0))
    return 1.0 + (1.0 / k) * math.log(p / (1.0 - p))


def classify(score):
    if score <= 40:
        return "FRIA"
    if score <= 74:
        return "MORNA"
    return "QUENTE"


def analyze_drivers(values, features, composite, comps, weights, score, CONFIG):
    metas = CONFIG["metas"]
    ratios = {"mencoes": features[0], "engajamentos": features[2], "sentimento": features[4], "brandfit": features[6]}
    contribs = {k: weights[k] * comps[k] for k in comps.keys()}
    contribs_sorted = sorted(contribs.items(), key=lambda x: x[1], reverse=True)

    ups = [k for k, _ in contribs_sorted if ratios[k] >= 1.0]
    downs = [k for k, _ in contribs_sorted if ratios[k] < 1.0]

    recs = []
    if score < 75:
        composite_target = score_to_composite_target(75.0, CONFIG)
        delta_needed = max(0.0, composite_target - composite)
        gaps = [(k, (1.0 - min(1.0, ratios[k])), weights[k]) for k in ratios if ratios[k] < 1.0]
        gaps_sorted = sorted(gaps, key=lambda x: (x[1] * x[2]), reverse=True)
        for k, _, _ in gaps_sorted[:2]:
            if k in ["mencoes", "engajamentos"]:
                recs.append(f"- Aumente **{k}** até ~{int(metas[k]):,} (alvo de meta).".replace(",", "."))
            elif k == "sentimento":
                recs.append(f"- Eleve **sentimento** para ≥ {metas['sentimento']:.0f} via criativos de valência positiva.")
            elif k == "brandfit":
                recs.append(f"- Suba **brandfit** para ≥ {metas['brandfit']:.1f} alinhando mensagem e territórios de marca.")
        if delta_needed > 0:
            recs.append(f"- Ganho composto necessário ~{delta_needed:.2f} para chegar a score ≈ 75.")
    else:
        recs.append("- **Manter pilares ≥ meta** e escalar formatos/canais vencedores.")
        if ratios["brandfit"] < 1.0:
            recs.append("- **Ajustar brandfit**: refine narrativa/CTAs para aderir mais ao território da marca.")

    txt = []
    if ups:
        txt.append("🔼 **Puxaram o score para cima:** " + ", ".join([u.capitalize() for u in ups]))
    if downs:
        txt.append("🔽 **Seguraram o score:** " + ", ".join([d.capitalize() for d in downs]))
    if recs:
        txt.append("🛠️ **Recomendações:**")
        txt.extend(recs)
    return "\n".join(txt) if txt else "Sem destaques relevantes; pilares próximos da meta."


def pct_of_goal(value, goal):
    try:
        return max(0.0, min(100.0, (float(value) / float(goal)) * 100.0))
    except Exception:
        return 0.0


# ---------- Visual helpers ----------
def big_score_donut_plotly(score: float, title: str = "Score"):
    try:
        import plotly.graph_objects as go
    except Exception:
        return None

    v = max(0, min(100, float(score)))
    fig = go.Figure(
        data=[
            go.Pie(
                values=[v, 100 - v],
                labels=[f"{v:.0f}%", ""],
                hole=0.7,
                textinfo="label",
                sort=False,
                direction="clockwise",
                showlegend=False,
            )
        ]
    )
    fig.update_traces(marker=dict(colors=["#5CB85C", "#E6E6E6"]))
    fig.update_layout(
        title={"text": title, "y": 0.92},
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
        width=420,
        annotations=[dict(text=f"{v:.0f}", x=0.5, y=0.5, font_size=42, showarrow=False)],
    )
    return fig


def to_png_bytes(fig) -> bytes:
    """
    Converte figura Plotly em PNG (requer `kaleido` instalado).
    """
    try:
        png = fig.to_image(format="png", scale=2)
        return png
    except Exception:
        return b""


# ---------- State & helpers ----------
def _normalize_weight_dict(w: Dict[str, float]) -> Dict[str, float]:
    arr = np.array([w["mencoes"], w["engajamentos"], w["sentimento"], w["brandfit"]], dtype=float)
    s = float(arr.sum()) or 1.0
    arr = arr / s
    return {"mencoes": float(arr[0]), "engajamentos": float(arr[1]), "sentimento": float(arr[2]), "brandfit": float(arr[3])}


@st.cache_data(show_spinner=False)
def parse_trends_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Guess date & series
    # If columns contain "date"/"data", treat as date
    date_col = None
    for c in df.columns:
        if str(c).strip().lower() in ["date", "data"]:
            date_col = c
            break
    if date_col is None:
        # fallback: first datetime-parsable column
        for c in df.columns:
            try:
                pd.to_datetime(df[c])
                date_col = c
                break
            except Exception:
                continue
    if date_col is None:
        raise ValueError("Não foi possível identificar a coluna de data. Use cabeçalho 'date' ou 'data'.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df.rename(columns={date_col: "date"})
    # Any other numeric columns are series
    value_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        # Maybe there's a 'volume' column not numeric (string); try coercion
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            value_cols = ["volume"]
    if not value_cols:
        raise ValueError("Nenhuma coluna numérica encontrada. Inclua 'volume' ou séries numéricas.")
    return df[["date"] + value_cols]


# =========================
# App
# =========================
st.set_page_config(page_title="Temperatura da Pauta (Pro)", page_icon="🔥", layout="wide")
st.title("🔥 Temperatura da Pauta — Pro")

with st.sidebar:
    st.header("Configurações")
    with st.expander("Metas (alvos)", expanded=True):
        meta_menc = st.number_input("Meta de Menções", min_value=0, value=int(DEFAULT_CONFIG["metas"]["mencoes"]), step=50)
        meta_eng = st.number_input("Meta de Engajamentos", min_value=0, value=int(DEFAULT_CONFIG["metas"]["engajamentos"]), step=250)
        meta_sent = st.number_input("Meta de Sentimento", min_value=0.0, max_value=100.0, value=float(DEFAULT_CONFIG["metas"]["sentimento"]), step=1.0)
        meta_bfit = st.number_input("Meta de Brandfit", min_value=0.0, max_value=10.0, value=float(DEFAULT_CONFIG["metas"]["brandfit"]), step=0.1)

    with st.expander("Pesos (importância relativa)", expanded=True):
        lock_norm = st.checkbox("Normalizar pesos para somar 100%", value=True, help="Quando ligado, os pesos são automaticamente normalizados.")
        peso_menc = st.number_input("Peso - Menções", value=float(DEFAULT_CONFIG["pesos_iniciais"]["mencoes"]), step=0.1, format="%.2f")
        peso_eng = st.number_input("Peso - Engajamentos", value=float(DEFAULT_CONFIG["pesos_iniciais"]["engajamentos"]), step=0.1, format="%.2f")
        peso_sent = st.number_input("Peso - Sentimento", value=float(DEFAULT_CONFIG["pesos_iniciais"]["sentimento"]), step=0.1, format="%.2f")
        peso_bfit = st.number_input("Peso - Brandfit", value=float(DEFAULT_CONFIG["pesos_iniciais"]["brandfit"]), step=0.1, format="%.2f")
        pesos_raw = {"mencoes": peso_menc, "engajamentos": peso_eng, "sentimento": peso_sent, "brandfit": peso_bfit}
        pesos_norm = _normalize_weight_dict(pesos_raw) if lock_norm else pesos_raw
        st.caption(f"Pesos efetivos (normalizados): mencões={pesos_norm['mencoes']:.2%}, engaj={pesos_norm['engajamentos']:.2%}, sent={pesos_norm['sentimento']:.2%}, brandfit={pesos_norm['brandfit']:.2%}")

    ratio_cap = st.slider("Cap de razão (limite superior)", 1.0, 5.0, float(DEFAULT_CONFIG["caps"]["ratio_cap"]), 0.1)
    log_k = st.slider("Inclinação logística (k)", 1.0, 8.0, float(DEFAULT_CONFIG["logistic_k"]), 0.5)

    # Persistência (salvar/carregar)
    st.markdown("---")
    st.subheader("Salvar / Carregar config")
    cfg = {
        "metas": {"mencoes": meta_menc, "engajamentos": meta_eng, "sentimento": meta_sent, "brandfit": meta_bfit},
        "pesos_iniciais": pesos_norm,
        "caps": {"ratio_cap": ratio_cap},
        "logistic_k": log_k,
    }
    cfg_json = json.dumps(cfg, ensure_ascii=False, indent=2)
    st.download_button("⬇️ Baixar config JSON", data=cfg_json, file_name="config_temperatura_pauta.json", mime="application/json")
    up = st.file_uploader("Carregar JSON de config", type=["json"], key="cfg_upload")
    if up is not None:
        try:
            loaded = json.load(up)
            st.session_state["loaded_cfg"] = loaded
            st.success("Config carregada! Reaplique os valores manualmente nos controles, se desejar.")
        except Exception as e:
            st.error(f"Falha ao carregar JSON: {e}")

CONFIG = {
    "metas": {"mencoes": meta_menc, "engajamentos": meta_eng, "sentimento": meta_sent, "brandfit": meta_bfit},
    "pesos_iniciais": pesos_norm,
    "caps": {"ratio_cap": ratio_cap},
    "logistic_k": log_k,
}

# ---- Entradas principais + cenário B para comparação
st.subheader("Entradas da Pauta")
col_a, col_b = st.columns(2)
with col_a:
    pauta = st.text_input("Nome da pauta", value="Minha Pauta")
    menc_ig = st.number_input("Menções no Instagram", min_value=0, value=120, step=10)
    eng_ig = st.number_input("Engajamentos no Instagram", min_value=0, value=3000, step=100)
with col_b:
    menc_tw = st.number_input("Menções no Twitter/X", min_value=0, value=80, step=10)
    eng_tw = st.number_input("Engajamentos no Twitter/X", min_value=0, value=1800, step=100)
    brand = st.number_input("Brandfit (0-10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    sent = st.number_input("Sentimento (0-100)", min_value=0.0, max_value=100.0, value=72.0, step=1.0)

# Cenário B (opcional)
with st.expander("Comparar com cenário B (opcional)"):
    enable_b = st.checkbox("Ativar cenário B", value=False)
    if enable_b:
        c1, c2 = st.columns(2)
        with c1:
            menc_ig_b = st.number_input("B: Menções Instagram", min_value=0, value=140, step=10)
            eng_ig_b = st.number_input("B: Engaj Instagram", min_value=0, value=3200, step=100)
            sent_b = st.number_input("B: Sentimento (0-100)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        with c2:
            menc_tw_b = st.number_input("B: Menções Twitter/X", min_value=0, value=90, step=10)
            eng_tw_b = st.number_input("B: Engaj Twitter/X", min_value=0, value=1900, step=100)
            brand_b = st.number_input("B: Brandfit (0-10)", min_value=0.0, max_value=10.0, value=7.6, step=0.1)

# ---- Cálculos
menc_total = menc_ig + menc_tw
eng_total = eng_ig + eng_tw

feats = normalize_features(menc_total, eng_total, sent, brand, CONFIG)
composite, comps, weights = compute_composite(feats, CONFIG)
score = composite_to_score(composite, CONFIG)
classe = classify(score)
values = {"mencoes": menc_total, "engajamentos": eng_total, "sentimento": sent, "brandfit": brand}
analise = analyze_drivers(values, feats, composite, comps, weights, score, CONFIG)

# Se houver cenário B, calcule também
if 'enable_b' in locals() and enable_b:
    menc_total_b = menc_ig_b + menc_tw_b
    eng_total_b = eng_ig_b + eng_tw_b
    feats_b = normalize_features(menc_total_b, eng_total_b, sent_b, brand_b, CONFIG)
    composite_b, comps_b, weights_b = compute_composite(feats_b, CONFIG)
    score_b = composite_to_score(composite_b, CONFIG)
    classe_b = classify(score_b)

# ---------- TABS ----------
tab_over, tab_drivers, tab_trends, tab_export = st.tabs(["Visão Geral", "Drivers", "Google Trends", "Exportar"])

with tab_over:
    st.markdown("---")
    # KPIs
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Score Final", f"{score:.1f}")
    kpi2.metric("Classificação", classe)
    kpi3.metric("Menções (IG+X)", f"{menc_total:,}".replace(",", "."))
    kpi4.metric("Engajamentos (IG+X)", f"{eng_total:,}".replace(",", "."))
    kpi5.metric("Sent/Brandfit", f"{sent:.0f} / {brand:.1f}")

    # Score Donut
    st.subheader("Score — Visão Geral")
    try:
        import plotly
        fig_big = big_score_donut_plotly(score, title=f"Score — {pauta}")
        st.plotly_chart(fig_big, use_container_width=False)
        png_bytes = to_png_bytes(fig_big)
        if png_bytes:
            st.download_button("Baixar gráfico (PNG)", data=png_bytes, file_name="score_donut.png")
    except Exception:
        st.info("Plotly indisponível; instale `plotly` e `kaleido` para download do gráfico.")

    # Progresso por Indicador
    st.subheader("Progresso em relação às metas")
    dados_progresso = pd.DataFrame(
        {
            "Indicador": ["Menções", "Engajamentos", "Sentimento", "Brandfit"],
            "Percentual da Meta (%)": [
                pct_of_goal(menc_total, meta_menc),
                pct_of_goal(eng_total, meta_eng),
                pct_of_goal(sent, meta_sent),
                pct_of_goal(brand, meta_bfit),
            ],
        }
    )
    st.bar_chart(dados_progresso.set_index("Indicador"))

    # Comparação (se ativada)
    if 'enable_b' in locals() and enable_b:
        st.subheader("Comparação A x B")
        delta = score_b - score
        colx, coly, colz = st.columns(3)
        colx.metric("Score A", f"{score:.1f}")
        coly.metric("Score B", f"{score_b:.1f}", delta=f"{delta:+.1f}")
        colz.metric("Classe B", classe_b)

with tab_drivers:
    st.subheader("Análise dos Drivers")
    st.markdown(analise)

    # Tabela de componentes e pesos
    st.subheader("Componentes & Pesos")
    df_comp = pd.DataFrame(
        {
            "Componente": ["Menções", "Engajamentos", "Sentimento", "Brandfit"],
            "Valor Composto": [comps["mencoes"], comps["engajamentos"], comps["sentimento"], comps["brandfit"]],
            "Peso Normalizado": [weights["mencoes"], weights["engajamentos"], weights["sentimento"], weights["brandfit"]],
        }
    )
    st.dataframe(df_comp.style.format({"Valor Composto": "{:.3f}", "Peso Normalizado": "{:.2%}"}), use_container_width=True)

with tab_trends:
    st.subheader("Google Trends — últimos 15 dias (ou mais)")
    st.caption("Envie um CSV com colunas: **date/data** e **volume** (ou múltiplas séries numéricas).")
    trends_file = st.file_uploader("CSV de Google Trends", type=["csv"], key="trends_upload")
    if trends_file is not None:
        try:
            df_trends = parse_trends_csv(trends_file)
            # Range temporal + rolling
            min_d, max_d = df_trends["date"].min(), df_trends["date"].max()
            d1, d2 = st.slider("Período", min_value=min_d.to_pydatetime(), max_value=max_d.to_pydatetime(), value=(max_d.to_pydatetime() - pd.Timedelta(days=14), max_d.to_pydatetime()))
            win = st.slider("Suavização (média móvel, dias)", 1, 7, 1)
            # Seleção de séries
            series_cols = [c for c in df_trends.columns if c != "date"]
            sel = st.multiselect("Séries", options=series_cols, default=series_cols[:1])
            plot_df = df_trends[(df_trends["date"] >= pd.to_datetime(d1)) & (df_trends["date"] <= pd.to_datetime(d2))].copy()
            if sel:
                plot_df = plot_df[["date"] + sel]
            else:
                st.warning("Selecione ao menos uma série para visualizar.")
            # Smoothing
            if win > 1 and sel:
                for c in sel:
                    plot_df[c] = plot_df[c].rolling(win, min_periods=1).mean()

            # Plot (Altair)
            try:
                import altair as alt

                df_long = plot_df.melt("date", var_name="Série", value_name="Volume")
                line = (
                    alt.Chart(df_long)
                    .mark_line()
                    .encode(
                        x=alt.X("date:T", title="Data"),
                        y=alt.Y("Volume:Q", title="Volume de pesquisas"),
                        color="Série:N",
                        tooltip=["date:T", "Série:N", "Volume:Q"],
                    )
                    .interactive()
                )
                st.altair_chart(line, use_container_width=True)
            except Exception:
                st.line_chart(plot_df.set_index("date"))

            with st.expander("Prévia dos dados"):
                st.dataframe(plot_df, use_container_width=True)

        except Exception as e:
            st.error(f"Falha ao ler/plotar Trends: {e}")
    else:
        st.info("Envie um CSV com 'date' (ou 'data') e uma ou mais colunas numéricas (ex.: 'volume', 'marca_1', 'marca_2').")

with tab_export:
    st.subheader("Exportar relatório simples")
    # CSV com resumo
    resumo = pd.DataFrame(
        {
            "Métrica": ["Score", "Classe", "Menções", "Engajamentos", "Sentimento", "Brandfit"],
            "Valor": [f"{score:.1f}", classe, menc_total, eng_total, sent, brand],
        }
    )
    csv_bytes = resumo.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar resumo (CSV)", data=csv_bytes, file_name="resumo_temperatura_pauta.csv", mime="text/csv")

st.markdown("---")
st.caption("Creative Data • Temperatura da Pauta — Pro")
