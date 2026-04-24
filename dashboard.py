"""
dashboard.py — Dashboard Streamlit · Trafic marchandises Port d'Abidjan
========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Port d'Abidjan — Prévisions Trafic",
    page_icon="🚢", layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "hw"    : "#D85A30",
    "sarima": "#1D9E75",
    "naif"  : "#888780",
    "reel"  : "#2C2C2A",
    "ic"    : "rgba(55,138,221,0.10)",
    "blue"  : "#378ADD",
}

MOIS_NOMS = ["Jan","Fév","Mar","Avr","Mai","Jun",
             "Jul","Aoû","Sep","Oct","Nov","Déc"]

AXE_TO_SEGS = {
    "sens"        : ["Import", "Export"],
    "composante"  : ["March. générales", "Prod. pétroliers", "Prod. de pêche"],
    "destination" : ["National", "Transit", "Transbordement"],
    "conteneur"   : ["Conteneurisé", "Non conteneurisé"],
}

SEG_TO_AXE = {
    "Import"           : "sens",
    "Export"           : "sens",
    "March. générales" : "composante",
    "Prod. pétroliers" : "composante",
    "Prod. de pêche"   : "composante",
    "National"         : "destination",
    "Transit"          : "destination",
    "Transbordement"   : "destination",
    "Conteneurisé"     : "conteneur",
    "Non conteneurisé" : "conteneur",
}

st.markdown("""
<style>
section[data-testid="stSidebar"]{background:#1a1a18;}
section[data-testid="stSidebar"] *{color:#e0dfd6 !important;}
.kpi{background:#f8f8f6;border-radius:10px;padding:14px 16px;
     text-align:center;margin-bottom:4px;}
.kpi-label{font-size:11px;color:#73726c;margin-bottom:4px;}
.kpi-value{font-size:22px;font-weight:500;color:#2c2c2a;}
.kpi-sub{font-size:11px;color:#888780;margin-top:2px;}
.warn{background:#faeeda;border-left:3px solid #BA7517;padding:8px 12px;
      border-radius:4px;font-size:12px;color:#633806;margin:6px 0;}
.badge{padding:3px 10px;border-radius:6px;font-size:11px;font-weight:500;}
.badge-td{background:rgba(29,158,117,0.12);color:#085041;}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────
@st.cache_data
def load_all():
    with open("forecasts.pkl","rb") as f:
        fc_data = pickle.load(f)
    with open("models.pkl","rb") as f:
        mdl_data = pickle.load(f)
    # Escales (optionnel)
    fc_esc, mdl_esc, ser_esc = None, None, None
    try:
        with open("forecasts_escales.pkl","rb") as f:
            esc_raw = pickle.load(f)
            fc_esc  = esc_raw['forecasts']
            mdl_esc = esc_raw['meta']
        with open("series_escales.pkl","rb") as f:
            ser_esc = pickle.load(f)
    except FileNotFoundError:
        pass
    # Conteneurs (optionnel)
    fc_cnt, mdl_cnt, ser_cnt = None, None, None
    try:
        with open("forecasts_conteneurs.pkl","rb") as f:
            cnt_raw = pickle.load(f)
            fc_cnt  = cnt_raw['forecasts']
            mdl_cnt = cnt_raw['meta']
        with open("series_conteneurs.pkl","rb") as f:
            ser_cnt = pickle.load(f)
    except FileNotFoundError:
        pass
    return fc_data, mdl_data, fc_esc, mdl_esc, ser_esc, fc_cnt, mdl_cnt, ser_cnt

try:
    fc_data, mdl_data, fc_esc, mdl_esc, ser_esc, fc_cnt, mdl_cnt, ser_cnt = load_all()
    forecasts      = fc_data["forecasts"]
    series_store   = fc_data["series_store"]
    meta           = fc_data["meta"]
    ANNEE_MIN_FC   = int(fc_data["annee_min_fc"])
    ANNEE_MAX_FC   = int(fc_data["annee_max_fc"])
    ANNEE_MAX_DATA = int(fc_data["annee_max_data"])
    fitted_models  = mdl_data["models"]
    SERIES_WARN    = meta.get("series_warn", {})
except FileNotFoundError as e:
    st.error(f"Fichier manquant : {e}\n\nLance d'abord :\n"
             "```\npython train_models.py\npython forecast_engine.py\n```")
    st.stop()

AXES_SEGS = {
    "Total"              : ["Total"],
    "Grandes composantes": ["March. générales","Prod. pétroliers","Prod. de pêche"],
    "Sens de trafic"     : ["Import","Export"],
    "Destination"        : ["National","Transit","Transbordement"],
    "Conteneurisation"   : ["Conteneurisé","Non conteneurisé"],
}

# ─────────────────────────────────────────────
# 2. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # ── LOGO + TITRE ────────────────────────────────────────────
    try:
        col_l, col_m, col_r = st.columns([1, 2, 1])
        with col_m:
            st.image("logo_PAA.jpg", use_container_width=True)
    except Exception:
        pass
    st.markdown(
        "<div style='text-align:center;font-weight:700;font-size:15px;"
        "color:#1a1a18;margin-top:4px;'>Port Autonome d'Abidjan</div>"
        "<div style='text-align:center;color:#888;font-size:11px;"
        "margin-top:2px;margin-bottom:4px;'>Tableau de bord des prévisions de trafic</div>",
        unsafe_allow_html=True)
    st.markdown("---")

    # ── SÉLECTEUR DE MODULE (radio horizontal) ─────────────────
    module = st.radio(
        "Module",
        ["📦 Marchses", "🚢 Escales", "📦 Conteneurs"],
        horizontal=True,
        label_visibility="collapsed",
        key="module_sel",
    )
    st.markdown("---")

    # ── PAGES DU MODULE SÉLECTIONNÉ ────────────────────────────
    if module == "📦 Marchses":
        page = st.radio("", [
            "KPIs globaux",
            "Analyse historique",
            "Prévisions court terme",
            "Prévisions long terme",
            "Analyse par axe",
        ], label_visibility="collapsed", key="page_march")
        approche_key = "top_down"; bu_axe = None
        axe_label = list(AXES_SEGS.keys())[0]
        segment   = AXES_SEGS[axe_label][0]
    elif module == "🚢 Escales":
        page = st.radio("", [
            "Escales — KPIs",
            "Escales — Historique",
            "Escales — Prévisions CT",
            "Escales — Prévisions LT",
            "Escales — Par terminal",
        ], label_visibility="collapsed", key="page_esc")
        approche_key = "top_down"; bu_axe = None
        axe_label = list(AXES_SEGS.keys())[0]
        segment   = AXES_SEGS[axe_label][0]

    else:  # Conteneurs
        page = st.radio("", [
            "Conteneurs — KPIs",
            "Conteneurs — Historique",
            "Conteneurs — Prévisions CT",
            "Conteneurs — Prévisions LT",
            "Conteneurs — Par segment",
        ], label_visibility="collapsed", key="page_cnt")
        approche_key = "top_down"; bu_axe = None
        axe_label = list(AXES_SEGS.keys())[0]
        segment   = AXES_SEGS[axe_label][0]

    # ── HORIZON (commun aux 3 modules) ─────────────────────────
    st.markdown("---")
    st.markdown("**Horizon de prévision**")
    horizon     = st.slider("Nombre d'années", min_value=1,
                            max_value=15, value=3, step=1,
                            key="horizon_global")
    annee_cible = ANNEE_MIN_FC + horizon - 1
    st.markdown(f"**{ANNEE_MIN_FC}** → **{annee_cible}**")

    # ── EXPORT EXCEL (commun aux 3 modules) ────────────────────
    st.markdown("---")
    st.markdown("**Export Excel**")
    try:
        from generate_tableau import generate_xlsx_long_terme, generate_xlsx_court_terme
        buf_lt = generate_xlsx_long_terme(
            forecasts=forecasts, series_store=series_store,
            annee_max_data=ANNEE_MAX_DATA,
            annee_min_fc=ANNEE_MIN_FC, horizon=horizon,
            approche_key=approche_key, bu_axe=bu_axe,
        )
        st.download_button(
            label=f"📥 Long terme ({ANNEE_MIN_FC}–{annee_cible})",
            data=buf_lt,
            file_name=f"PAA_previsions_long_terme_{ANNEE_MIN_FC}_{annee_cible}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        buf_ct = generate_xlsx_court_terme(
            forecasts=forecasts, series_store=series_store,
            annee_max_data=ANNEE_MAX_DATA, annee_fc=ANNEE_MIN_FC,
            approche_key=approche_key, bu_axe=bu_axe,
        )
        st.download_button(
            label=f"📥 Court terme ({ANNEE_MIN_FC})",
            data=buf_ct,
            file_name=f"PAA_previsions_court_terme_{ANNEE_MIN_FC}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Export indisponible : {e}")

    # ── ADMIN (discret, tout en bas) ────────────────────────────
    st.markdown("---")
    if st.button("⚙️ Administration", use_container_width=True,
                 type="secondary", key="btn_admin"):
        st.session_state["show_admin"] = not st.session_state.get("show_admin", False)

# ─────────────────────────────────────────────
# 3. HELPERS RÉCONCILIATION
# ─────────────────────────────────────────────
def get_serie(seg):
    return series_store.get(seg)

def _get_parts(annee):
    """Parts annuelles de chaque segment dans le Total."""
    tot = series_store.get("Total")
    if tot is None: return {}
    total_ann = float(tot[tot.index.year == annee].sum())
    if total_ann == 0: return {}
    parts = {}
    for seg, axe in SEG_TO_AXE.items():
        s = series_store.get(seg)
        if s is not None:
            parts[seg] = float(s[s.index.year == annee].sum()) / total_ann
        else:
            parts[seg] = 1.0 / len(AXE_TO_SEGS[axe])
    return parts

def ann(seg, yr):
    """Valeur annuelle selon l'approche de réconciliation choisie."""
    fc = forecasts.get((seg, yr), {})
    if approche_key == "top_down":
        if seg == "Total":
            return fc.get("annuel", 0)
        axe = SEG_TO_AXE.get(seg)
        if axe:
            td = fc.get("annuel_td", {}).get(axe)
            if td is not None: return round(float(td), 3)
        return fc.get("annuel", 0)
    # Bottom-up
    segs_axe = AXE_TO_SEGS.get(bu_axe, [])
    if seg == "Total":
        return round(sum(
            forecasts.get((s, yr), {}).get("annuel", 0)
            for s in segs_axe), 3)
    if SEG_TO_AXE.get(seg) == bu_axe:
        return fc.get("annuel", 0)
    # Hors axe : redistribution
    total_bu = sum(forecasts.get((s, yr), {}).get("annuel", 0) for s in segs_axe)
    parts    = _get_parts(yr - 1)
    return round(total_bu * parts.get(seg, 0), 3)

def mens(seg, yr):
    """Array 12 valeurs mensuelles selon l'approche choisie."""
    fc = forecasts.get((seg, yr), {})
    if approche_key == "top_down":
        if seg == "Total":
            return fc.get("fc", np.zeros(12))
        axe = SEG_TO_AXE.get(seg)
        if axe:
            td = fc.get("top_down", {}).get(axe)
            if td is not None: return np.array(td)
        return fc.get("fc", np.zeros(12))
    # Bottom-up
    segs_axe = AXE_TO_SEGS.get(bu_axe, [])
    if seg == "Total":
        arrays = [forecasts.get((s, yr), {}).get("fc", np.zeros(12)) for s in segs_axe]
        return np.sum(arrays, axis=0) if arrays else np.zeros(12)
    if SEG_TO_AXE.get(seg) == bu_axe:
        return fc.get("fc", np.zeros(12))
    arrays   = [forecasts.get((s, yr), {}).get("fc", np.zeros(12)) for s in segs_axe]
    total_bu = np.sum(arrays, axis=0) if arrays else np.zeros(12)
    parts    = _get_parts(yr - 1)
    return np.array([v * parts.get(seg, 0) for v in total_bu])

def show_warn(seg):
    if seg in SERIES_WARN:
        st.markdown(f'<div class="warn">⚠️ {SERIES_WARN[seg]}</div>',
                    unsafe_allow_html=True)

def show_badge():
    st.markdown('<span class="badge badge-td">Top-down : Trafic global × parts N-1</span>',
                unsafe_allow_html=True)

def kpi(col, label, value, sub=""):
    col.markdown(f"""<div class="kpi">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div></div>""",
        unsafe_allow_html=True)

def plo(fig, title, h=380):
    fig.update_layout(
        title=title, height=h,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0ee"),
        yaxis=dict(gridcolor="#f0f0ee"),
        legend=dict(orientation="h", y=1.08, font=dict(size=10)),
        hovermode="x unified",
        margin=dict(t=60,b=40,l=60,r=20))
    return fig

# ─────────────────────────────────────────────
# PAGE 1 : KPIs GLOBAUX
# ─────────────────────────────────────────────
if page == "KPIs globaux":
    st.markdown("## 📦 Marchandises — KPIs globaux")
    show_badge()

    s_tot = get_serie("Total")
    ann_r = s_tot.groupby(s_tot.index.year).sum()
    cagr  = (ann_r.iloc[-1]/ann_r.iloc[0])**(1/(len(ann_r)-1))-1

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1,f"Tonnage {ANNEE_MAX_DATA}",
        f"{ann_r[ANNEE_MAX_DATA]:.1f} Mt","année complète")
    kpi(c2,f"CAGR 2015–{ANNEE_MAX_DATA}",
        f"+{100*cagr:.1f}%","croissance annuelle moy.")

    tot_next = ann("Total", ANNEE_MIN_FC)
    if tot_next:
        croiss = 100*(tot_next/ann_r[ANNEE_MAX_DATA]-1)
        kpi(c3,f"Prévu {ANNEE_MIN_FC}",f"{tot_next:.1f} Mt",
            f"{croiss:+.1f}% vs {ANNEE_MAX_DATA}")
    tot_cible = ann("Total", annee_cible)
    if tot_cible:
        kpi(c4,f"Prévu {annee_cible}",f"{tot_cible:.1f} Mt",
            f"horizon +{annee_cible-ANNEE_MAX_DATA} ans")

    st.markdown("###")
    fig = go.Figure()
    fig.add_bar(x=ann_r.index.tolist(), y=ann_r.values,
                name="Réel", marker_color=COLORS["blue"], opacity=0.85)
    fc_pts = [(yr, ann("Total", yr))
              for yr in range(ANNEE_MIN_FC, annee_cible+1)
              if ann("Total", yr)]
    if fc_pts:
        yrs, vals = zip(*fc_pts)
        fig.add_bar(x=list(yrs), y=list(vals),
                    name="Prévu", marker_color=COLORS["hw"], opacity=0.75)
    st.plotly_chart(
        plo(fig,"Tonnage annuel (Mt) — réel + prévisions"),
        use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        si = get_serie("Import"); se = get_serie("Export")
        ai = si[si.index.year==ANNEE_MAX_DATA].sum()
        ae = se[se.index.year==ANNEE_MAX_DATA].sum()
        fig2 = go.Figure(go.Pie(
            labels=["Import","Export"], values=[ai,ae],
            marker_colors=[COLORS["blue"],COLORS["sarima"]],
            hole=0.4, textinfo="label+percent"))
        fig2.update_layout(title=f"Import vs Export ({ANNEE_MAX_DATA})",
            height=300, plot_bgcolor="white",
            paper_bgcolor="white", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        sd = ["National","Transit","Transbordement"]
        vd = [get_serie(s)[get_serie(s).index.year==ANNEE_MAX_DATA].sum()
              for s in sd]
        fig3 = go.Figure(go.Bar(y=sd, x=vd, orientation="h",
            marker_color=[COLORS["blue"],COLORS["sarima"],COLORS["hw"]],
            opacity=0.85))
        fig3.update_layout(title=f"Par destination ({ANNEE_MAX_DATA})",
            height=300, plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#f0f0ee"))
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 2 : ANALYSE HISTORIQUE
# ─────────────────────────────────────────────
elif page == "Analyse historique":
    st.markdown(f"## 📦 Marchandises — Analyse historique · {segment}")
    show_warn(segment)
    serie = get_serie(segment)
    if serie is None:
        st.error("Série non disponible"); st.stop()

    trend = serie.rolling(12, center=True).mean()
    fig = go.Figure()
    fig.add_scatter(x=serie.index, y=serie.values, mode="lines",
                    name="Mensuel",
                    line=dict(color=COLORS["blue"],width=1.2), opacity=0.7)
    fig.add_scatter(x=trend.index, y=trend.values, mode="lines",
                    name="Tendance MM12",
                    line=dict(color=COLORS["hw"],width=2,dash="dash"))
    fig.add_vrect(x0="2020-01-01",x1="2020-12-31",
                  fillcolor="#BA7517", opacity=0.07,
                  annotation_text="Covid")
    st.plotly_chart(plo(fig,f"Série mensuelle — {segment}"),
                    use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        sai = serie.groupby(serie.index.month).mean()
        fig_s = go.Figure(go.Bar(x=MOIS_NOMS, y=sai.values,
            marker_color=["#BA7517" if v==sai.max() else
                          "#888780" if v==sai.min() else
                          COLORS["blue"] for v in sai.values], opacity=0.85))
        fig_s.add_hline(y=sai.mean(), line_dash="dash",
                        line_color=COLORS["hw"])
        fig_s.update_layout(title="Saisonnalité moyenne", height=300,
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#f0f0ee"))
        st.plotly_chart(fig_s, use_container_width=True)
    with col2:
        dfh = pd.DataFrame({"v":serie.values,
                            "a":serie.index.year,
                            "m":serie.index.month})
        piv = dfh.pivot(index="m", columns="a", values="v")
        fig_h = go.Figure(go.Heatmap(
            z=piv.values, x=[str(c) for c in piv.columns],
            y=MOIS_NOMS, colorscale="YlOrRd", showscale=True))
        fig_h.update_layout(title="Heatmap mensuelle", height=300,
            plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_h, use_container_width=True)

    ann_s = serie.groupby(serie.index.year).sum()
    fig_a = go.Figure(go.Bar(
        x=ann_s.index.tolist(), y=ann_s.values,
        marker_color=COLORS["blue"], opacity=0.85,
        text=[f"{v:.1f}" for v in ann_s.values],
        textposition="outside"))
    fig_a.update_layout(title="Tonnage annuel (Mt)", height=300,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#f0f0ee"))
    st.plotly_chart(fig_a, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3 : PRÉVISIONS COURT TERME
# ─────────────────────────────────────────────
elif page == "Prévisions court terme":
    st.markdown(f"## 📦 Marchandises — Prévisions court terme · {segment} · {ANNEE_MIN_FC}")
    show_badge()
    show_warn(segment)

    fc_raw = forecasts.get((segment, ANNEE_MIN_FC))
    if fc_raw is None:
        st.warning("Prévisions non disponibles."); st.stop()

    fc_vals  = mens(segment, ANNEE_MIN_FC)
    tot_ann  = ann(segment, ANNEE_MIN_FC) or 0
    fc_dates = fc_raw["dates"]
    ic_lo    = fc_raw.get("ic_lo", np.zeros(12))
    ic_hi    = fc_raw.get("ic_hi", np.zeros(12))

    serie = get_serie(segment)
    reel  = serie[serie.index.year==ANNEE_MIN_FC] if serie is not None else None
    mdl   = fitted_models.get(segment, {})
    wmape = mdl.get("wmape_test")
    mtype = mdl.get("type","—")

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1,f"Total prévu {ANNEE_MIN_FC}",f"{tot_ann:.2f} Mt",mtype.upper())
    kpi(c2,"WMAPE test 2025",
        f"{wmape:.1f}%" if wmape else "N/A","performance historique")
    kpi(c3,"Mois max prévu",
        MOIS_NOMS[int(np.argmax(fc_vals))],
        f"{float(np.max(fc_vals)):.3f} Mt")
    kpi(c4,"Mois min prévu",
        MOIS_NOMS[int(np.argmin(fc_vals))],
        f"{float(np.min(fc_vals)):.3f} Mt")
    st.markdown("###")

    fig = go.Figure()
    fig.add_scatter(
        x=list(fc_dates)+list(fc_dates[::-1]),
        y=list(ic_hi)+list(ic_lo[::-1]),
        fill="toself", fillcolor=COLORS["ic"],
        line=dict(color="rgba(255,255,255,0)"), name="IC 95%")
    color_fc = COLORS.get(mtype, COLORS["blue"])
    fig.add_scatter(x=fc_dates, y=fc_vals,
                    mode="lines+markers", name="Prévu",
                    line=dict(color=color_fc, width=2.5),
                    marker=dict(size=5))
    if reel is not None and len(reel) > 0:
        fig.add_scatter(x=reel.index, y=reel.values,
                        mode="lines+markers", name="Réel",
                        line=dict(color=COLORS["reel"], width=2.5),
                        marker=dict(size=6))
    st.plotly_chart(
        plo(fig,f"Prévisions mensuelles {ANNEE_MIN_FC} — {segment}"),
        use_container_width=True)

    with st.expander("Détail mensuel"):
        rows = []
        for i,(d,v) in enumerate(zip(fc_dates, fc_vals)):
            row = {
                "Mois"        : d.strftime("%b %Y"),
                "Prévu (Mt)"  : round(float(v),3),
                "IC bas (Mt)" : round(float(ic_lo[i]),3),
                "IC haut (Mt)": round(float(ic_hi[i]),3),
            }
            if reel is not None and len(reel)==12:
                row["Réel (Mt)"]  = round(float(reel.values[i]),3)
                row["Écart (Mt)"] = round(float(reel.values[i]-v),3)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# PAGE 4 : PRÉVISIONS LONG TERME
# ─────────────────────────────────────────────
elif page == "Prévisions long terme":
    st.markdown(f"## 📦 Marchandises — Prévisions long terme · {segment} → {annee_cible}")
    show_badge()
    show_warn(segment)

    serie = get_serie(segment)
    mdl   = fitted_models.get(segment, {})
    wmape = mdl.get("wmape_test")

    fc_ann_d = {yr: ann(segment, yr)
                for yr in range(ANNEE_MIN_FC, annee_cible+1)
                if ann(segment, yr) is not None}

    if not fc_ann_d:
        st.warning("Aucune prévision disponible."); st.stop()

    yr_list   = sorted(fc_ann_d.keys())
    reel_last = (float(serie[serie.index.year==ANNEE_MAX_DATA].sum())
                 if serie is not None else None)
    fc_next   = fc_ann_d.get(ANNEE_MIN_FC)
    croiss    = 100*(fc_next/reel_last-1) if fc_next and reel_last else None

    c1,c2,c3 = st.columns(3)
    kpi(c1,f"Prévu {ANNEE_MIN_FC}",f"{fc_ann_d[ANNEE_MIN_FC]:.1f} Mt",
        f"{croiss:+.1f}% vs {ANNEE_MAX_DATA}" if croiss else "")
    kpi(c2,f"Prévu {annee_cible}",f"{fc_ann_d[annee_cible]:.1f} Mt",
        f"horizon +{annee_cible-ANNEE_MAX_DATA} ans")
    kpi(c3,"Modèle",mdl.get("type","—").upper(),"prévision récursive")
    st.markdown("###")

    fig = go.Figure()
    if serie is not None:
        ann_r2 = serie.resample("A").sum()
        fig.add_scatter(
            x=ann_r2.index.year.tolist(), y=ann_r2.values,
            mode="lines+markers", name="Réel",
            line=dict(color=COLORS["reel"],width=2.5),
            marker=dict(size=6))
    fig.add_scatter(
        x=yr_list, y=[fc_ann_d[y] for y in yr_list],
        mode="lines+markers", name="Prévu",
        line=dict(color=COLORS["hw"],width=2.5), marker=dict(size=5))
    if wmape:
        ic_hi2 = [fc_ann_d[y]*(1+(wmape/100)*(y-ANNEE_MAX_DATA)**0.5*0.3)
                  for y in yr_list]
        ic_lo2 = [max(0,fc_ann_d[y]*(1-(wmape/100)*(y-ANNEE_MAX_DATA)**0.5*0.3))
                  for y in yr_list]
        fig.add_scatter(
            x=yr_list+yr_list[::-1], y=ic_hi2+ic_lo2[::-1],
            fill="toself", fillcolor="rgba(216,90,48,0.08)",
            line=dict(color="rgba(255,255,255,0)"), name="IC (croissant)")
    fig.add_vline(x=ANNEE_MAX_DATA+0.5, line_dash="dot",
                  line_color="#888780",
                  annotation_text="Début prévision")
    st.plotly_chart(
        plo(fig,f"Prévisions long terme (Mt/an) — {segment}"),
        use_container_width=True)

    with st.expander("Tableau annuel complet"):
        rows = [{"Année":y,
                 "Prévu (Mt)": round(fc_ann_d[y],2),
                 "Croissance": f"{100*(fc_ann_d[y]/fc_ann_d.get(y-1,fc_ann_d[y])-1):+.1f}%"
                               if y>yr_list[0] else "—"}
                for y in yr_list]
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PAGE 6 : ANALYSE PAR AXE
# ─────────────────────────────────────────────
elif page == "Analyse par axe":
    # Sélecteurs inline (axe + série)
    col_ax, col_yr = st.columns([2, 1])
    with col_ax:
        axe_label = st.selectbox(
            "Axe d'analyse",
            list(AXES_SEGS.keys()),
            key="axe_label_inline"
        )
    with col_yr:
        annee_cible_axe = st.selectbox(
            "Année de prévision",
            list(range(ANNEE_MIN_FC, ANNEE_MAX_FC + 1)),
            key="annee_axe_inline"
        )
    annee_cible = annee_cible_axe

    st.markdown(f"## 📦 Marchandises — Analyse par axe · {axe_label} · {annee_cible}")
    show_badge()

    segs     = AXES_SEGS[axe_label]
    cols_axe = ["#378ADD","#1D9E75","#D85A30","#534AB7","#BA7517","#888780"]

    fig = go.Figure()
    for i, seg in enumerate(segs):
        fc_raw = forecasts.get((seg, annee_cible))
        if fc_raw is None: continue
        fc_vals = mens(seg, annee_cible)
        fig.add_scatter(
            x=[d.strftime("%b") for d in fc_raw["dates"]],
            y=fc_vals,
            mode="lines+markers", name=seg,
            line=dict(color=cols_axe[i%len(cols_axe)], width=2),
            marker=dict(size=4))
    st.plotly_chart(
        plo(fig,f"Prévisions mensuelles {annee_cible} — {axe_label}"),
        use_container_width=True)

    rows = []
    for seg in segs:
        s   = get_serie(seg)
        rl  = (float(s[s.index.year==ANNEE_MAX_DATA].sum())
               if s is not None else None)
        tot = ann(seg, annee_cible)
        mdl = fitted_models.get(seg, {})
        if tot is not None:
            croiss = 100*(tot/rl-1) if rl and rl>0 else None
            rows.append({
                "Segment"                     : seg,
                f"Réel {ANNEE_MAX_DATA} (Mt)" : round(rl,2) if rl else "—",
                f"Prévu {annee_cible} (Mt)"   : round(tot,2),
                "Croissance"                  : f"{croiss:+.1f}%" if croiss else "—",
                "WMAPE test"                  : f"{mdl.get('wmape_test','—'):.1f}%"
                                                if mdl.get("wmape_test") else "—",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True, hide_index=True)
    if len(segs) > 1:
        tots = {s: ann(s, annee_cible)
                for s in segs if ann(s, annee_cible) is not None}
        if tots:
            fig2 = go.Figure(go.Pie(
                labels=list(tots.keys()), values=list(tots.values()),
                marker_colors=cols_axe[:len(tots)],
                hole=0.4, textinfo="label+percent"))
            fig2.update_layout(
                title=f"Répartition {annee_cible} — {axe_label}",
                height=320, plot_bgcolor="white",
                paper_bgcolor="white", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGES ESCALES
# ═══════════════════════════════════════════════════════════════

def page_escales_indisponible():
    st.warning("Fichiers escales non trouvés. Lance d'abord :\n"
               "`python train_models_escales.py`\n"
               "`python forecast_engine_escales.py`")

if page == "── Escales ──":
    st.info("Sélectionne une sous-page Escales dans le menu.")

elif page == "Escales — KPIs":
    if fc_esc is None:
        page_escales_indisponible()
    else:
        st.markdown("## 🚢 Escales — Indicateurs clés")
        SEGS_ESC = mdl_esc['segs']
        ann_hist = mdl_esc['ann_total_hist']
        yr_last  = mdl_esc['annee_fin']
        yr_prev  = yr_last + 1

        tot_last = ann_hist[yr_last]
        tot_prev = fc_esc[yr_prev]['annuel']
        cagr_5   = (ann_hist[yr_last] / ann_hist[yr_last - 5]) ** (1/5) - 1
        cagr_10  = (ann_hist[yr_last] / ann_hist[yr_last - 10]) ** (1/10) - 1
        tot_2030 = fc_esc[2030]['annuel']
        tot_2040 = fc_esc[2040]['annuel']

        c1, c2, c3, c4 = st.columns(4)
        for col_kpi, label, val, sub in [
            (c1, f"Réalisé {yr_last}", f"{tot_last:,}", "escales"),
            (c2, f"Prévu {yr_prev}",   f"{tot_prev:,}", f"IC95%: {fc_esc[yr_prev]['ic_lo']:,}–{fc_esc[yr_prev]['ic_hi']:,}"),
            (c3, "CAGR 5 ans",         f"{cagr_5*100:+.1f}%", f"{yr_last-5}–{yr_last}"),
            (c4, "Prévu 2040",         f"{tot_2040:,}", f"IC95%: {fc_esc[2040]['ic_lo']:,}–{fc_esc[2040]['ic_hi']:,}"),
        ]:
            col_kpi.markdown(
                f'<div class="kpi"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{val}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True)

        st.markdown("---")
        col_g, col_t = st.columns([2, 1])

        with col_g:
            st.markdown("#### Evolution du total des escales")
            years_h = list(ann_hist.keys())
            vals_h  = list(ann_hist.values())
            years_p = list(range(yr_last + 1, 2041))
            vals_p  = [fc_esc[y]['annuel'] for y in years_p]
            lo_p    = [fc_esc[y]['ic_lo']  for y in years_p]
            hi_p    = [fc_esc[y]['ic_hi']  for y in years_p]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years_h, y=vals_h, name="Réalisé",
                line=dict(color="#2C2C2A", width=2.5),
                mode="lines+markers", marker=dict(size=6)))
            fig.add_trace(go.Scatter(
                x=[yr_last] + years_p,
                y=[vals_h[-1]] + vals_p,
                name="Prévu", line=dict(color="#378ADD", width=2, dash="dash"),
                mode="lines+markers", marker=dict(size=5)))
            fig.add_trace(go.Scatter(
                x=years_p + years_p[::-1],
                y=hi_p + lo_p[::-1],
                fill="toself", fillcolor="rgba(55,138,221,0.10)",
                line=dict(width=0), name="IC 95%", showlegend=True))
            fig.update_layout(
                height=340, plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=-0.15),
                margin=dict(l=40, r=20, t=20, b=40),
                yaxis=dict(title="Nombre d'escales", gridcolor="#ECECEC"),
                xaxis=dict(gridcolor="#ECECEC"))
            st.plotly_chart(fig, use_container_width=True)

        with col_t:
            st.markdown("#### Parts par terminal (2025)")
            parts_2025 = mdl_esc['parts_terminaux'][yr_last]
            rows_kpi = sorted(parts_2025.items(), key=lambda x: -x[1])
            tbl = pd.DataFrame(rows_kpi, columns=["Terminal", "Part 2025 (%)"])
            tbl["Part 2025 (%)"] = tbl["Part 2025 (%)"].map(lambda x: f"{x:.1f}%")
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Qualité du modèle")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Modèle", "Holt amortie")
        cc2.metric("WMAPE (LOO-CV)", f"{mdl_esc['wmape']:.1f}%")
        cc3.metric("RMSE", f"{mdl_esc['rmse']:.0f} escales/an")
        st.caption(f"Paramètres optimaux : α={mdl_esc['alpha']:.2f}  "
                   f"β={mdl_esc['beta']:.2f}  φ={mdl_esc['phi']:.2f}  |  "
                   f"Profil saisonnier : moyenne {mdl_esc['annees_profil'][0]}–{mdl_esc['annees_profil'][-1]}  |  "
                   f"Top-down : clés N-1 récursif")

elif page == "Escales — Historique":
    if fc_esc is None or ser_esc is None:
        page_escales_indisponible()
    else:
        st.markdown("## 🚢 Escales — Analyse historique")
        SEGS_ESC = mdl_esc['segs']
        ann_hist = mdl_esc['ann_total_hist']
        yr_last  = mdl_esc['annee_fin']
        yr_debut = mdl_esc['annee_debut']

        tab1, tab2, tab3 = st.tabs(["Série mensuelle", "Profil saisonnier", "Parts par terminal"])

        with tab1:
            seg_sel = st.selectbox("Terminal", ["Total"] + SEGS_ESC)
            s = ser_esc[seg_sel]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values,
                name=seg_sel, line=dict(color="#378ADD", width=1.8),
                fill="tozeroy", fillcolor="rgba(55,138,221,0.08)"))
            fig.update_layout(
                height=320, plot_bgcolor="white", paper_bgcolor="white",
                title=f"Série mensuelle — {seg_sel}",
                yaxis=dict(title="Nombre d'escales", gridcolor="#ECECEC"),
                xaxis=dict(gridcolor="#ECECEC"),
                margin=dict(l=40, r=20, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("#### Profil saisonnier moyen (2023-2025)")
            profil = mdl_esc['profil_saisonnier']
            noms_m = mdl_esc['noms_mois']
            cols_bar = ["#2196F3" if i != profil.argmax() else "#D85A30"
                        for i in range(12)]
            fig2 = go.Figure(go.Bar(
                x=noms_m, y=profil,
                marker_color=cols_bar,
                text=[f"{v:.1f}%" for v in profil],
                textposition="outside"))
            fig2.update_layout(
                height=300, plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Part mensuelle (%)", gridcolor="#ECECEC"),
                margin=dict(l=40, r=20, t=20, b=30))
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Profil calculé sur 2023-2025 (années sans données manquantes). "
                       "Décembre est le mois le plus actif (9.5%), juin le plus calme (7.7%).")

        with tab3:
            st.markdown("#### Evolution des parts par terminal (% du total annuel)")
            parts_h = mdl_esc['parts_terminaux']
            years_h = list(ann_hist.keys())
            COLORS_ESC = ["#2196F3","#FF9800","#4CAF50","#9C27B0","#F44336",
                          "#00BCD4","#795548","#607D8B","#E91E63","#FF5722"]
            fig3 = go.Figure()
            for i, g in enumerate(SEGS_ESC):
                vals = [parts_h[yr][g] for yr in years_h]
                fig3.add_trace(go.Scatter(
                    x=years_h, y=vals, name=g,
                    line=dict(color=COLORS_ESC[i % len(COLORS_ESC)], width=1.8),
                    mode="lines+markers", marker=dict(size=5)))
            fig3.update_layout(
                height=370, plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Part (%)", gridcolor="#ECECEC"),
                xaxis=dict(gridcolor="#ECECEC"),
                legend=dict(orientation="h", y=-0.3),
                margin=dict(l=40, r=20, t=20, b=80))
            st.plotly_chart(fig3, use_container_width=True)

elif page == "Escales — Prévisions CT":
    if fc_esc is None:
        page_escales_indisponible()
    else:
        st.markdown("## 🚢 Escales — Prévisions court terme (2026)")
        yr_last  = mdl_esc['annee_fin']
        yr_prev  = yr_last + 1
        noms_m   = mdl_esc['noms_mois']
        SEGS_ESC = mdl_esc['segs']

        seg_ct = st.selectbox("Terminal", ["Total"] + SEGS_ESC, key="esc_ct_seg")

        # Données 2025 réel + 2026 prévu
        if seg_ct == "Total":
            mens_reel = fc_esc[('historique', yr_last)]['mensuel']
            mens_prev = fc_esc[yr_prev]['mensuel']
            mens_lo   = fc_esc[yr_prev]['mensuel_lo']
            mens_hi   = fc_esc[yr_prev]['mensuel_hi']
            ann_reel  = fc_esc[('historique', yr_last)]['annuel']
            ann_prev  = fc_esc[yr_prev]['annuel']
        else:
            mens_reel = fc_esc[('historique', yr_last)]['segments'][seg_ct]
            mens_prev = fc_esc[yr_prev]['segments'][seg_ct]
            mens_lo   = np.round(mens_prev * 0.90).astype(int)
            mens_hi   = np.round(mens_prev * 1.10).astype(int)
            ann_reel  = int(mens_reel.sum())
            ann_prev  = int(mens_prev.sum())

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Réalisé {yr_last}", f"{ann_reel:,}", "escales")
        c2.metric(f"Prévu {yr_prev}",   f"{ann_prev:,}",
                  f"{(ann_prev-ann_reel)/ann_reel*100:+.1f}%")
        if seg_ct == "Total":
            c3.metric("IC 95%",
                      f"{fc_esc[yr_prev]['ic_lo']:,} – {fc_esc[yr_prev]['ic_hi']:,}")

        st.markdown("---")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=noms_m, y=mens_reel,
            name=f"Réalisé {yr_last}",
            marker_color="rgba(44,44,42,0.7)"))
        fig.add_trace(go.Bar(
            x=noms_m, y=mens_prev,
            name=f"Prévu {yr_prev}",
            marker_color="rgba(55,138,221,0.85)"))
        if seg_ct == "Total":
            fig.add_trace(go.Scatter(
                x=noms_m + noms_m[::-1],
                y=list(mens_hi) + list(mens_lo[::-1]),
                fill="toself", fillcolor="rgba(55,138,221,0.12)",
                line=dict(width=0), name="IC 95%"))
        fig.update_layout(
            barmode="group", height=340,
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.2),
            yaxis=dict(title="Nombre d'escales", gridcolor="#ECECEC"),
            margin=dict(l=40, r=20, t=20, b=50))
        st.plotly_chart(fig, use_container_width=True)

        # Tableau mensuel
        rows_ct = []
        for m in range(12):
            rows_ct.append({
                "Mois"          : noms_m[m],
                f"Réalisé {yr_last}": int(mens_reel[m]),
                f"Prévu {yr_prev}"  : int(mens_prev[m]),
                "Variation"     : f"{(mens_prev[m]-mens_reel[m])/max(mens_reel[m],1)*100:+.1f}%"
            })
        rows_ct.append({
            "Mois"          : "TOTAL ANNUEL",
            f"Réalisé {yr_last}": ann_reel,
            f"Prévu {yr_prev}"  : ann_prev,
            "Variation"     : f"{(ann_prev-ann_reel)/ann_reel*100:+.1f}%"
        })
        st.dataframe(pd.DataFrame(rows_ct), use_container_width=True, hide_index=True)

elif page == "Escales — Prévisions LT":
    if fc_esc is None:
        page_escales_indisponible()
    else:
        st.markdown("## 🚢 Escales — Prévisions long terme")
        yr_last  = mdl_esc['annee_fin']
        yr_debut = mdl_esc['annee_debut']
        SEGS_ESC = mdl_esc['segs']
        ann_hist = mdl_esc['ann_total_hist']

        annee_fin_lt = yr_last + horizon

        years_h = list(range(yr_debut, yr_last + 1))
        years_p = list(range(yr_last + 1, annee_fin_lt + 1))
        vals_h  = [ann_hist[y] for y in years_h]
        vals_p  = [fc_esc[y]['annuel'] for y in years_p]
        lo_p    = [fc_esc[y]['ic_lo']  for y in years_p]
        hi_p    = [fc_esc[y]['ic_hi']  for y in years_p]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years_h, y=vals_h, name="Réalisé",
            line=dict(color="#2C2C2A", width=2.5),
            mode="lines+markers", marker=dict(size=6)))
        fig.add_trace(go.Scatter(
            x=[yr_last] + years_p,
            y=[vals_h[-1]] + vals_p,
            name="Prévu (Holt amortie)",
            line=dict(color="#378ADD", width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=5)))
        fig.add_trace(go.Scatter(
            x=years_p + years_p[::-1],
            y=hi_p + lo_p[::-1],
            fill="toself", fillcolor="rgba(55,138,221,0.10)",
            line=dict(width=0), name="IC 95%"))
        fig.update_layout(
            height=360, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.18),
            yaxis=dict(title="Nombre d'escales", gridcolor="#ECECEC"),
            xaxis=dict(gridcolor="#ECECEC"),
            margin=dict(l=40, r=20, t=20, b=50))
        st.plotly_chart(fig, use_container_width=True)

        # Tableau long terme
        rows_lt = []
        prev_v  = None
        for yr in years_h + years_p:
            if yr <= yr_last:
                v = ann_hist[yr]
                typ = "Réalisé"
                ic  = "—"
            else:
                v   = fc_esc[yr]['annuel']
                typ = "Prévu"
                ic  = f"{fc_esc[yr]['ic_lo']:,} – {fc_esc[yr]['ic_hi']:,}"
            cro = f"{(v-prev_v)/prev_v*100:+.1f}%" if prev_v else "—"
            rows_lt.append({"Année": yr, "Type": typ,
                            "Total escales": f"{v:,}", "Croissance": cro, "IC 95%": ic})
            prev_v = v
        st.dataframe(pd.DataFrame(rows_lt), use_container_width=True, hide_index=True)

elif page == "Escales — Par terminal":
    if fc_esc is None:
        page_escales_indisponible()
    else:
        st.markdown("## 🚢 Escales — Répartition par terminal")
        yr_last  = mdl_esc['annee_fin']
        SEGS_ESC = mdl_esc['segs']
        ann_hist = mdl_esc['ann_total_hist']

        col_a, col_b = st.columns([1, 2])
        with col_a:
            annee_sel = st.selectbox(
                "Année",
                list(range(yr_last, 2041)),
                key="esc_term_yr")

        COLORS_ESC = ["#2196F3","#FF9800","#4CAF50","#9C27B0","#F44336",
                      "#00BCD4","#795548","#607D8B","#E91E63","#FF5722"]

        is_hist = annee_sel <= yr_last

        if is_hist:
            segs_vals = {g: int(fc_esc[('historique', annee_sel)]['segments'][g].sum())
                         for g in SEGS_ESC}
            tot_sel = ann_hist[annee_sel]
        else:
            segs_vals = {g: int(fc_esc[annee_sel]['segments'][g].sum())
                         for g in SEGS_ESC}
            tot_sel = fc_esc[annee_sel]['annuel']

        col_pie, col_bar = st.columns(2)

        with col_pie:
            fig_pie = go.Figure(go.Pie(
                labels=list(segs_vals.keys()),
                values=list(segs_vals.values()),
                marker_colors=COLORS_ESC,
                hole=0.4,
                textinfo="label+percent",
                textfont=dict(size=10)))
            fig_pie.update_layout(
                height=360,
                title=f"Répartition {annee_sel} ({'Réalisé' if is_hist else 'Prévu'})",
                paper_bgcolor="white",
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            # Evolution annuelle par terminal (empilé)
            years_all = list(range(yr_last - 4, annee_sel + 1))
            fig_bar = go.Figure()
            for i, g in enumerate(SEGS_ESC):
                vals_g = []
                for y in years_all:
                    if y <= yr_last:
                        vals_g.append(int(fc_esc[('historique', y)]['segments'][g].sum()))
                    else:
                        vals_g.append(int(fc_esc[y]['segments'][g].sum()))
                fig_bar.add_trace(go.Bar(
                    x=years_all, y=vals_g, name=g,
                    marker_color=COLORS_ESC[i % len(COLORS_ESC)]))
            fig_bar.update_layout(
                barmode="stack", height=360,
                plot_bgcolor="white", paper_bgcolor="white",
                title=f"Empilé par terminal ({years_all[0]}–{annee_sel})",
                legend=dict(orientation="h", y=-0.35, font=dict(size=9)),
                yaxis=dict(title="Escales", gridcolor="#ECECEC"),
                margin=dict(l=40, r=10, t=40, b=80))
            st.plotly_chart(fig_bar, use_container_width=True)

        # Tableau détail
        st.markdown("---")
        rows_t = []
        for g in SEGS_ESC:
            v = segs_vals[g]
            p = v / tot_sel * 100 if tot_sel > 0 else 0
            rows_t.append({"Terminal": g,
                            f"Escales {annee_sel}": f"{v:,}",
                            "Part (%)": f"{p:.1f}%"})
        rows_t.append({"Terminal": "TOTAL",
                        f"Escales {annee_sel}": f"{tot_sel:,}",
                        "Part (%)": "100.0%"})
        st.dataframe(pd.DataFrame(rows_t), use_container_width=True, hide_index=True)
        if not is_hist:
            st.caption(f"IC 95% total : {fc_esc[annee_sel]['ic_lo']:,} – "
                       f"{fc_esc[annee_sel]['ic_hi']:,} escales  |  "
                       f"Clé utilisée : parts {fc_esc[annee_sel]['cle_annee']}")


# ═══════════════════════════════════════════════════════════════
# PAGES CONTENEURS
# ═══════════════════════════════════════════════════════════════

def page_cnt_indisponible():
    st.warning("Fichiers conteneurs non trouvés. Lance d'abord :\n"
               "`python train_models_conteneurs.py`\n"
               "`python forecast_engine_conteneurs.py`")

if page == "── Conteneurs ──":
    st.info("Sélectionne une sous-page Conteneurs dans le menu.")

elif page == "Conteneurs — KPIs":
    if fc_cnt is None:
        page_cnt_indisponible()
    else:
        st.markdown("## 📦 Conteneurs — Indicateurs clés")
        ann_hist = mdl_cnt['ann_total_hist']
        yr_last  = mdl_cnt['annee_fin']
        yr_prev  = yr_last + 1

        tot_last = ann_hist[yr_last]
        tot_prev = fc_cnt[yr_prev]['annuel']
        cagr_5   = (ann_hist[yr_last] / ann_hist[yr_last-5]) ** (1/5) - 1
        tot_2030 = fc_cnt[2030]['annuel']
        tot_2040 = fc_cnt[2040]['annuel']

        c1, c2, c3, c4 = st.columns(4)
        for col_k, label, val, sub in [
            (c1, f"Réalisé {yr_last}",  f"{tot_last:,}",  "TEU"),
            (c2, f"Prévu {yr_prev}",    f"{tot_prev:,}",
             f"IC95%: {fc_cnt[yr_prev]['ic_lo']:,}–{fc_cnt[yr_prev]['ic_hi']:,}"),
            (c3, "CAGR 5 ans",          f"{cagr_5*100:+.1f}%", f"{yr_last-5}–{yr_last}"),
            (c4, "Prévu 2040",          f"{tot_2040:,}",
             f"IC95%: {fc_cnt[2040]['ic_lo']:,}–{fc_cnt[2040]['ic_hi']:,}"),
        ]:
            col_k.markdown(
                f'<div class="kpi"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{val}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True)

        st.markdown("---")
        col_g, col_t = st.columns([2,1])
        with col_g:
            st.markdown("#### Évolution du total conteneurs (TEU)")
            years_h = list(ann_hist.keys())
            vals_h  = list(ann_hist.values())
            years_p = list(range(yr_last+1, 2041))
            vals_p  = [fc_cnt[y]['annuel'] for y in years_p]
            lo_p    = [fc_cnt[y]['ic_lo']  for y in years_p]
            hi_p    = [fc_cnt[y]['ic_hi']  for y in years_p]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years_h, y=vals_h, name="Réalisé",
                line=dict(color="#2C2C2A", width=2.5), mode="lines+markers",
                marker=dict(size=6)))
            fig.add_trace(go.Scatter(
                x=[yr_last]+years_p, y=[vals_h[-1]]+vals_p,
                name="Prévu", line=dict(color="#378ADD", width=2, dash="dash"),
                mode="lines+markers", marker=dict(size=5)))
            fig.add_trace(go.Scatter(
                x=years_p+years_p[::-1], y=hi_p+lo_p[::-1],
                fill="toself", fillcolor="rgba(55,138,221,0.10)",
                line=dict(width=0), name="IC 95%"))
            fig.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=-0.15),
                margin=dict(l=40,r=20,t=20,b=40),
                yaxis=dict(title="TEU", gridcolor="#ECECEC"),
                xaxis=dict(gridcolor="#ECECEC"))
            st.plotly_chart(fig, use_container_width=True)

        with col_t:
            st.markdown("#### Parts par terminal (2025)")
            pts = mdl_cnt['parts_term'][yr_last]
            rows = sorted(pts.items(), key=lambda x: -x[1])
            tbl = pd.DataFrame(rows, columns=["Terminal","Part (%)"])
            tbl["Part (%)"] = tbl["Part (%)"].map(lambda x: f"{x:.1f}%")
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Qualité des modèles")
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Total", f"err={mdl_cnt['err_tot']:.1f}%", "Holt amortie 2023-2025")
        cc2.metric("Non transbordé", f"WMAPE={mdl_cnt['wmape_nt']:.1f}%", "Holt amortie")
        cc3.metric("Transbordé TC2", "Constant 2025", f"{mdl_cnt['transb_tc2_2025']:,} TEU/an")
        cc4.metric("Transbordé hab.", f"WMAPE={mdl_cnt['wmape_th']:.1f}%", "Holt amortie")

elif page == "Conteneurs — Historique":
    if fc_cnt is None or ser_cnt is None:
        page_cnt_indisponible()
    else:
        st.markdown("## 📦 Conteneurs — Analyse historique")
        ann_hist = mdl_cnt['ann_total_hist']
        yr_last  = mdl_cnt['annee_fin']
        SEGS_T   = mdl_cnt['segs_term']
        COLORS_C = ["#2196F3","#FF9800","#4CAF50","#9C27B0","#F44336"]

        tab1, tab2, tab3 = st.tabs(["Série mensuelle", "Profil saisonnier", "Parts par segment"])

        with tab1:
            seg_sel = st.selectbox("Série", ["Total"]+SEGS_T+["Non transb.","Transbordé","Transb. TC2","Transb. habituel"])
            s = ser_cnt[seg_sel]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=seg_sel,
                line=dict(color="#378ADD", width=1.8),
                fill="tozeroy", fillcolor="rgba(55,138,221,0.08)"))
            fig.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
                title=f"Série mensuelle — {seg_sel} (TEU)",
                yaxis=dict(title="TEU", gridcolor="#ECECEC"),
                margin=dict(l=40,r=20,t=40,b=30))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"⚠️ Rupture structurelle en 2023 due au démarrage du TC2 et à l'explosion du transbordement.")

        with tab2:
            st.markdown("#### Profil saisonnier moyen (2023-2025)")
            profil = mdl_cnt['profil_saisonnier']
            noms_m = mdl_cnt['noms_mois']
            cols_bar = ["#2196F3" if i!=np.argmax(profil) else "#D85A30" for i in range(12)]
            fig2 = go.Figure(go.Bar(x=noms_m, y=profil, marker_color=cols_bar,
                text=[f"{v:.1f}%" for v in profil], textposition="outside"))
            fig2.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Part mensuelle (%)", gridcolor="#ECECEC"),
                margin=dict(l=40,r=20,t=20,b=30))
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.markdown("#### Évolution des parts par terminal (%)")
            pts_h = mdl_cnt['parts_term']
            years_h = list(ann_hist.keys())
            fig3 = go.Figure()
            for i, g in enumerate(SEGS_T):
                vals = [pts_h[yr][g] for yr in years_h]
                fig3.add_trace(go.Scatter(x=years_h, y=vals, name=g,
                    line=dict(color=COLORS_C[i%len(COLORS_C)], width=1.8),
                    mode="lines+markers", marker=dict(size=5)))
            fig3.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Part (%)", gridcolor="#ECECEC"),
                legend=dict(orientation="h", y=-0.3),
                margin=dict(l=40,r=20,t=20,b=80))
            st.plotly_chart(fig3, use_container_width=True)

elif page == "Conteneurs — Prévisions CT":
    if fc_cnt is None:
        page_cnt_indisponible()
    else:
        st.markdown("## 📦 Conteneurs — Prévisions court terme (2026)")
        yr_last  = mdl_cnt['annee_fin']
        yr_prev  = yr_last + 1
        noms_m   = mdl_cnt['noms_mois']
        SEGS_T   = mdl_cnt['segs_term']

        seg_ct = st.selectbox("Série", ["Total"]+SEGS_T+["Non transb.","Transbordé","Transb. TC2","Transb. habituel"],
                              key="cnt_ct_seg")

        if seg_ct == "Total":
            mens_r = fc_cnt[('historique', yr_last)]['mensuel']
            mens_p = fc_cnt[yr_prev]['mensuel']
            mens_lo= fc_cnt[yr_prev]['mensuel_lo']
            mens_hi= fc_cnt[yr_prev]['mensuel_hi']
            ann_r  = fc_cnt[('historique', yr_last)]['annuel']
            ann_p  = fc_cnt[yr_prev]['annuel']
        elif seg_ct in SEGS_T:
            mens_r = fc_cnt[('historique', yr_last)]['segments_term'][seg_ct]
            mens_p = fc_cnt[yr_prev]['segments_term'][seg_ct]
            mens_lo= np.round(mens_p * 0.90).astype(int)
            mens_hi= np.round(mens_p * 1.10).astype(int)
            ann_r  = int(mens_r.sum())
            ann_p  = int(mens_p.sum())
        else:
            mens_r = fc_cnt[('historique', yr_last)]['segments_dest'][seg_ct]
            mens_p = fc_cnt[yr_prev]['segments_dest'][seg_ct]
            mens_lo= np.round(mens_p * 0.90).astype(int)
            mens_hi= np.round(mens_p * 1.10).astype(int)
            ann_r  = int(mens_r.sum())
            ann_p  = int(mens_p.sum())

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Réalisé {yr_last}", f"{ann_r:,}", "TEU")
        c2.metric(f"Prévu {yr_prev}", f"{ann_p:,}",
                  f"{(ann_p-ann_r)/ann_r*100:+.1f}%" if ann_r else "")
        if seg_ct == "Total":
            c3.metric("IC 95%", f"{fc_cnt[yr_prev]['ic_lo']:,} – {fc_cnt[yr_prev]['ic_hi']:,}")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=noms_m, y=mens_r, name=f"Réalisé {yr_last}",
            marker_color="rgba(44,44,42,0.7)"))
        fig.add_trace(go.Bar(x=noms_m, y=mens_p, name=f"Prévu {yr_prev}",
            marker_color="rgba(55,138,221,0.85)"))
        if seg_ct == "Total":
            fig.add_trace(go.Scatter(x=noms_m+noms_m[::-1],
                y=list(mens_hi)+list(mens_lo[::-1]),
                fill="toself", fillcolor="rgba(55,138,221,0.12)",
                line=dict(width=0), name="IC 95%"))
        fig.update_layout(barmode="group", height=340,
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.2),
            yaxis=dict(title="TEU", gridcolor="#ECECEC"),
            margin=dict(l=40,r=20,t=20,b=50))
        st.plotly_chart(fig, use_container_width=True)

        rows_ct = []
        for m in range(12):
            rows_ct.append({"Mois": noms_m[m],
                f"Réalisé {yr_last}": f"{int(mens_r[m]):,}",
                f"Prévu {yr_prev}":   f"{int(mens_p[m]):,}",
                "Variation": f"{(mens_p[m]-mens_r[m])/max(mens_r[m],1)*100:+.1f}%"})
        rows_ct.append({"Mois": "TOTAL ANNUEL",
            f"Réalisé {yr_last}": f"{ann_r:,}",
            f"Prévu {yr_prev}":   f"{ann_p:,}",
            "Variation": f"{(ann_p-ann_r)/ann_r*100:+.1f}%" if ann_r else "—"})
        st.dataframe(pd.DataFrame(rows_ct), use_container_width=True, hide_index=True)

elif page == "Conteneurs — Prévisions LT":
    if fc_cnt is None:
        page_cnt_indisponible()
    else:
        st.markdown("## 📦 Conteneurs — Prévisions long terme")
        yr_last  = mdl_cnt['annee_fin']
        ann_hist = mdl_cnt['ann_total_hist']

        annee_fin_lt = yr_last + horizon

        years_h = list(range(mdl_cnt['annee_debut'], yr_last+1))
        years_p = list(range(yr_last+1, annee_fin_lt+1))
        vals_h  = [ann_hist[y] for y in years_h]
        vals_p  = [fc_cnt[y]['annuel'] for y in years_p]
        lo_p    = [fc_cnt[y]['ic_lo']  for y in years_p]
        hi_p    = [fc_cnt[y]['ic_hi']  for y in years_p]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years_h, y=vals_h, name="Réalisé",
            line=dict(color="#2C2C2A", width=2.5), mode="lines+markers",
            marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=[yr_last]+years_p, y=[vals_h[-1]]+vals_p,
            name="Prévu (Holt amortie)",
            line=dict(color="#378ADD", width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=years_p+years_p[::-1], y=hi_p+lo_p[::-1],
            fill="toself", fillcolor="rgba(55,138,221,0.10)",
            line=dict(width=0), name="IC 95%"))
        fig.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.18),
            yaxis=dict(title="TEU", gridcolor="#ECECEC"),
            xaxis=dict(gridcolor="#ECECEC"),
            margin=dict(l=40,r=20,t=20,b=50))
        st.plotly_chart(fig, use_container_width=True)

        rows_lt = []
        prev_v  = None
        for yr in years_h + years_p:
            v   = ann_hist[yr] if yr <= yr_last else fc_cnt[yr]['annuel']
            typ = "Réalisé" if yr <= yr_last else "Prévu"
            ic  = "—" if yr <= yr_last else f"{fc_cnt[yr]['ic_lo']:,} – {fc_cnt[yr]['ic_hi']:,}"
            cro = f"{(v-prev_v)/prev_v*100:+.1f}%" if prev_v else "—"
            rows_lt.append({"Année": yr, "Type": typ,
                "Total (TEU)": f"{v:,}", "Croissance": cro, "IC 95%": ic})
            prev_v = v
        st.dataframe(pd.DataFrame(rows_lt), use_container_width=True, hide_index=True)

elif page == "Conteneurs — Par segment":
    if fc_cnt is None:
        page_cnt_indisponible()
    else:
        st.markdown("## 📦 Conteneurs — Répartition par segment")
        yr_last  = mdl_cnt['annee_fin']
        ann_hist = mdl_cnt['ann_total_hist']
        SEGS_T   = mdl_cnt['segs_term']
        COLORS_C = ["#2196F3","#FF9800","#4CAF50","#9C27B0","#F44336"]

        col_a, col_b = st.columns([1,2])
        with col_a:
            annee_sel = st.selectbox("Année",
                list(range(yr_last, 2041)), key="cnt_seg_yr")
            axe_sel = st.radio("Axe", ["Terminal","Destination"], key="cnt_seg_axe")

        is_hist = annee_sel <= yr_last

        if axe_sel == "Terminal":
            if is_hist:
                segs_v = {g: int(fc_cnt[('historique',annee_sel)]['segments_term'][g].sum())
                          for g in SEGS_T}
            else:
                segs_v = {g: int(fc_cnt[annee_sel]['segments_term'][g].sum())
                          for g in SEGS_T}
            segs_labels = SEGS_T
            colors = COLORS_C
        else:
            SEGS_D = ['Non transb.','Transb. TC2','Transb. habituel']
            if is_hist:
                segs_v = {
                    'Non transb.'     : int(fc_cnt[('historique',annee_sel)]['segments_dest']['Non transb.'].sum()),
                    'Transb. TC2'     : int(fc_cnt[('historique',annee_sel)]['segments_dest']['Transb. TC2'].sum()),
                    'Transb. habituel': int(fc_cnt[('historique',annee_sel)]['segments_dest']['Transb. habituel'].sum()),
                }
            else:
                segs_v = {
                    'Non transb.'     : int(fc_cnt[annee_sel]['segments_dest']['Non transb.'].sum()),
                    'Transb. TC2'     : int(fc_cnt[annee_sel]['segments_dest']['Transb. TC2'].sum()),
                    'Transb. habituel': int(fc_cnt[annee_sel]['segments_dest']['Transb. habituel'].sum()),
                }
            segs_labels = SEGS_D
            colors = ["#156082","#E97132","#A02B93"]

        tot_sel = ann_hist[annee_sel] if is_hist else fc_cnt[annee_sel]['annuel']

        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig_pie = go.Figure(go.Pie(
                labels=list(segs_v.keys()), values=list(segs_v.values()),
                marker_colors=colors, hole=0.4,
                textinfo="label+percent", textfont=dict(size=10)))
            fig_pie.update_layout(height=360,
                title=f"Répartition {annee_sel} ({'Réalisé' if is_hist else 'Prévu'}) — axe {axe_sel}",
                paper_bgcolor="white", showlegend=False,
                margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            years_all = list(range(max(yr_last-4, mdl_cnt['annee_debut']), annee_sel+1))
            fig_bar = go.Figure()
            for i, g in enumerate(segs_v.keys()):
                vals_g = []
                for y in years_all:
                    if y <= yr_last:
                        key_d = 'segments_term' if axe_sel=="Terminal" else 'segments_dest'
                        vals_g.append(int(fc_cnt[('historique',y)][key_d][g].sum()))
                    else:
                        key_d = 'segments_term' if axe_sel=="Terminal" else 'segments_dest'
                        vals_g.append(int(fc_cnt[y][key_d][g].sum()))
                fig_bar.add_trace(go.Bar(x=years_all, y=vals_g, name=g,
                    marker_color=colors[i%len(colors)]))
            fig_bar.update_layout(barmode="stack", height=360,
                plot_bgcolor="white", paper_bgcolor="white",
                title=f"Empilé ({years_all[0]}–{annee_sel})",
                legend=dict(orientation="h", y=-0.35, font=dict(size=9)),
                yaxis=dict(title="TEU", gridcolor="#ECECEC"),
                margin=dict(l=40,r=10,t=40,b=80))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        rows_t = []
        for g, v in segs_v.items():
            p = v/tot_sel*100 if tot_sel else 0
            rows_t.append({"Segment": g, f"TEU {annee_sel}": f"{v:,}", "Part (%)": f"{p:.1f}%"})
        rows_t.append({"Segment": "TOTAL", f"TEU {annee_sel}": f"{tot_sel:,}", "Part (%)": "100.0%"})
        st.dataframe(pd.DataFrame(rows_t), use_container_width=True, hide_index=True)
        if not is_hist:
            st.caption(f"IC 95% total : {fc_cnt[annee_sel]['ic_lo']:,} – "
                       f"{fc_cnt[annee_sel]['ic_hi']:,} TEU  |  "
                       f"Clé utilisée : parts {fc_cnt[annee_sel]['cle_annee']}")


# ═══════════════════════════════════════════════════════════════
# PAGE ADMINISTRATION
# ═══════════════════════════════════════════════════════════════

if st.session_state.get("show_admin", False):

    st.markdown("---")
    st.markdown("## ⚙️ Administration — Mise à jour des données")

    # ── AUTHENTIFICATION ─────────────────────────────────────────
    if not st.session_state.get("admin_auth", False):
        st.markdown("### 🔐 Accès restreint")
        pwd = st.text_input("Mot de passe", type="password", key="admin_pwd_input")
        if st.button("Se connecter", key="btn_login"):
            try:
                correct = st.secrets["ADMIN_PASSWORD"]
            except Exception:
                correct = "paa2025"   # fallback local pour tests
            if pwd == correct:
                st.session_state["admin_auth"] = True
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")
        st.stop()

    # ── INTERFACE ADMIN (après auth) ─────────────────────────────
    col_logout, _ = st.columns([1, 4])
    with col_logout:
        if st.button("🚪 Déconnexion", key="btn_logout"):
            st.session_state["admin_auth"] = False
            st.session_state["show_admin"] = False
            st.rerun()

    st.success("✅ Connecté — accès administrateur")
    st.markdown("---")

    # ── UPLOAD DES FICHIERS ──────────────────────────────────────
    st.markdown("### 📤 Charger les nouvelles données")
    st.caption("Chargez uniquement les fichiers disponibles. Les autres restent inchangés.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📦 Marchandises**")
        f_march = st.file_uploader("Fichier Excel marchandises",
                                   type=["xlsx"], key="up_march",
                                   label_visibility="collapsed")
        if f_march: st.success(f"✓ {f_march.name}")

    with col2:
        st.markdown("**🚢 Escales**")
        f_esc = st.file_uploader("Fichier Excel escales",
                                 type=["xlsx"], key="up_esc",
                                 label_visibility="collapsed")
        if f_esc: st.success(f"✓ {f_esc.name}")

    with col3:
        st.markdown("**📦 Conteneurs**")
        f_cnt = st.file_uploader("Fichier Excel conteneurs",
                                 type=["xlsx"], key="up_cnt",
                                 label_visibility="collapsed")
        if f_cnt: st.success(f"✓ {f_cnt.name}")

    if not any([f_march, f_esc, f_cnt]):
        st.info("Aucun fichier chargé. Veuillez uploader au moins un fichier Excel.")
    else:
        st.markdown("---")
        st.markdown("### 🔄 Lancer les prévisions")

        if st.button("🚀 Lancer la mise à jour", type="primary",
                     use_container_width=True, key="btn_run"):

            import tempfile, os, subprocess, json
            from datetime import datetime

            log     = []
            success = []
            errors  = []

            prog = st.progress(0, text="Démarrage...")
            status_box = st.empty()

            try:
                # ── GITHUB CONFIG ───────────────────────────────
                try:
                    gh_token = st.secrets["GITHUB_TOKEN"]
                    gh_repo  = st.secrets["GITHUB_REPO"]
                    use_gh   = True
                except Exception:
                    use_gh = False
                    st.warning("⚠️ GitHub non configuré — les pkl ne seront pas sauvegardés.")

                step = [0]
                total_steps = sum([
                    bool(f_march) * 2,
                    bool(f_esc)   * 2,
                    bool(f_cnt)   * 2,
                    1,  # journal
                ]) + (3 if use_gh else 0)

                def advance(msg):
                    step[0] += 1
                    pct = min(int(step[0] / max(total_steps, 1) * 100), 99)
                    prog.progress(pct, text=msg)
                    status_box.info(f"⏳ {msg}")
                    log.append(msg)

                # ── SAUVEGARDER LES FICHIERS UPLOADÉS ───────────
                tmpdir = tempfile.mkdtemp()

                if f_march:
                    path_march = os.path.join(tmpdir, "data_marchandises.xlsx")
                    with open(path_march, "wb") as fout:
                        fout.write(f_march.read())
                    # Copier avec le nom attendu par train_models.py
                    import shutil
                    shutil.copy(path_march,
                        "Jeu de donnée pour prevision PC08042026.xlsx")

                if f_esc:
                    path_esc = os.path.join(tmpdir, "data_escales.xlsx")
                    with open(path_esc, "wb") as fout:
                        fout.write(f_esc.read())
                    shutil.copy(path_esc, "data_prevision_Escale.xlsx")

                if f_cnt:
                    path_cnt = os.path.join(tmpdir, "data_conteneurs.xlsx")
                    with open(path_cnt, "wb") as fout:
                        fout.write(f_cnt.read())
                    shutil.copy(path_cnt, "data_prevision_Conteneur.xlsx")

                # ── ENTRAÎNEMENT MARCHANDISES ────────────────────
                if f_march:
                    advance("Entraînement modèles marchandises...")
                    r = subprocess.run(
                        ["python", "train_models.py"],
                        capture_output=True, text=True, timeout=300)
                    if r.returncode == 0:
                        success.append("Marchandises — modèles entraînés")
                    else:
                        errors.append(f"Marchandises train: {r.stderr[:200]}")

                    advance("Prévisions marchandises...")
                    r = subprocess.run(
                        ["python", "forecast_engine.py"],
                        capture_output=True, text=True, timeout=300)
                    if r.returncode == 0:
                        success.append("Marchandises — prévisions générées")
                    else:
                        errors.append(f"Marchandises forecast: {r.stderr[:200]}")

                # ── ENTRAÎNEMENT ESCALES ─────────────────────────
                if f_esc:
                    advance("Entraînement modèles escales...")
                    r = subprocess.run(
                        ["python", "train_models_escales.py"],
                        capture_output=True, text=True, timeout=300)
                    if r.returncode == 0:
                        success.append("Escales — modèles entraînés")
                    else:
                        errors.append(f"Escales train: {r.stderr[:200]}")

                    advance("Prévisions escales...")
                    r = subprocess.run(
                        ["python", "forecast_engine_escales.py"],
                        capture_output=True, text=True, timeout=300)
                    if r.returncode == 0:
                        success.append("Escales — prévisions générées")
                    else:
                        errors.append(f"Escales forecast: {r.stderr[:200]}")

                # ── ENTRAÎNEMENT CONTENEURS ──────────────────────
                if f_cnt:
                    advance("Entraînement modèles conteneurs...")
                    r = subprocess.run(
                        ["python", "train_models_conteneurs.py"],
                        capture_output=True, text=True, timeout=300)
                    if r.returncode == 0:
                        success.append("Conteneurs — modèles entraînés")
                    else:
                        errors.append(f"Conteneurs train: {r.stderr[:200]}")

                    advance("Prévisions conteneurs...")
                    r = subprocess.run(
                        ["python", "forecast_engine_conteneurs.py"],
                        capture_output=True, text=True, timeout=300)
                    if r.returncode == 0:
                        success.append("Conteneurs — prévisions générées")
                    else:
                        errors.append(f"Conteneurs forecast: {r.stderr[:200]}")

                # ── JOURNAL ──────────────────────────────────────
                advance("Mise à jour du journal...")
                hist_file = "historique.json"
                try:
                    with open(hist_file, "r", encoding="utf-8") as fh:
                        historique = json.load(fh)
                except Exception:
                    historique = []

                entry = {
                    "date"     : datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "modules"  : ([("Marchandises" if f_march else None),
                                   ("Escales"      if f_esc   else None),
                                   ("Conteneurs"   if f_cnt   else None)]),
                    "fichiers" : ([f_march.name if f_march else None,
                                   f_esc.name   if f_esc   else None,
                                   f_cnt.name   if f_cnt   else None]),
                    "statut"   : "OK" if not errors else "ERREURS",
                    "details"  : success + errors,
                }
                entry["modules"]  = [m for m in entry["modules"]  if m]
                entry["fichiers"] = [f for f in entry["fichiers"] if f]
                historique.insert(0, entry)

                with open(hist_file, "w", encoding="utf-8") as fh:
                    json.dump(historique, fh, ensure_ascii=False, indent=2)

                # ── COMMIT GITHUB ────────────────────────────────
                if use_gh and not errors:
                    advance("Sauvegarde sur GitHub...")
                    try:
                        import base64, urllib.request

                        pkl_files = [
                            "forecasts.pkl", "models.pkl", "series.pkl",
                            "forecasts_escales.pkl", "models_escales.pkl",
                            "series_escales.pkl",
                            "forecasts_conteneurs.pkl", "models_conteneurs.pkl",
                            "series_conteneurs.pkl",
                            "historique.json",
                        ]

                        headers = {
                            "Authorization": f"token {gh_token}",
                            "Accept": "application/vnd.github.v3+json",
                            "Content-Type": "application/json",
                        }

                        gh_ok = 0
                        for fname in pkl_files:
                            if not os.path.exists(fname):
                                continue
                            with open(fname, "rb") as fbin:
                                content_b64 = base64.b64encode(
                                    fbin.read()).decode()

                            # Récupérer le SHA actuel (nécessaire pour update)
                            api_url = (f"https://api.github.com/repos/"
                                       f"{gh_repo}/contents/{fname}")
                            req_get = urllib.request.Request(
                                api_url, headers=headers)
                            try:
                                with urllib.request.urlopen(req_get) as resp:
                                    sha = json.loads(resp.read())["sha"]
                            except Exception:
                                sha = None

                            payload = json.dumps({
                                "message": (f"[Admin] Mise à jour {fname} — "
                                            f"{entry['date']}"),
                                "content": content_b64,
                                **({"sha": sha} if sha else {}),
                            }).encode()

                            req_put = urllib.request.Request(
                                api_url, data=payload, headers=headers,
                                method="PUT")
                            try:
                                with urllib.request.urlopen(req_put):
                                    gh_ok += 1
                            except Exception as e_gh:
                                errors.append(f"GitHub {fname}: {str(e_gh)[:100]}")

                        if gh_ok > 0:
                            success.append(
                                f"GitHub — {gh_ok} fichiers sauvegardés")
                    except Exception as e_gh_main:
                        errors.append(f"GitHub: {str(e_gh_main)[:200]}")

                # ── RÉSULTAT FINAL ───────────────────────────────
                prog.progress(100, text="Terminé !")
                status_box.empty()

                if not errors:
                    st.success(
                        f"✅ Mise à jour réussie ! "
                        f"{len(success)} opération(s) effectuée(s).")
                    for s in success:
                        st.markdown(f"  • {s}")
                    st.info("🔄 Rechargez la page pour voir les nouvelles prévisions.")
                else:
                    st.warning(f"⚠️ Terminé avec {len(errors)} erreur(s).")
                    for s in success:
                        st.markdown(f"  ✓ {s}")
                    for e_msg in errors:
                        st.error(f"  ✗ {e_msg}")

            except Exception as e_global:
                prog.progress(0, text="Erreur")
                st.error(f"Erreur inattendue : {e_global}")

    # ── JOURNAL DES MISES À JOUR ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Journal des mises à jour")

    import json as _json
    try:
        with open("historique.json", "r", encoding="utf-8") as fh:
            hist = _json.load(fh)
        if not hist:
            st.info("Aucune mise à jour enregistrée.")
        else:
            rows = []
            for h in hist:
                rows.append({
                    "Date"    : h.get("date", "—"),
                    "Modules" : ", ".join(h.get("modules", [])),
                    "Fichiers": ", ".join(h.get("fichiers", [])),
                    "Statut"  : h.get("statut", "—"),
                })
            st.dataframe(pd.DataFrame(rows),
                         use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.info("Aucune mise à jour enregistrée.")
    except Exception as e_hist:
        st.warning(f"Impossible de lire le journal : {e_hist}")
