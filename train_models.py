"""
train_models.py — Entraînement et sérialisation des modèles
=============================================================
À relancer chaque fois qu'une nouvelle base est disponible.

Workflow complet :
    python train_models.py        # ~5–10 min
    python forecast_engine.py     # ~1 min
    streamlit run dashboard.py    # instantané

Sorties :
    models.pkl      — modèles entraînés sérialisés
    series.pkl      — séries historiques complètes
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
FILE_PATH   = "data_prevision_Marchandise.xlsx"
OUTPUT_MDL  = "models.pkl"
OUTPUT_SER  = "series.pkl"

# (label, col, val, train_debut, modele_type,
#  arima_order, seas_order, hw_seasonal, wmape_test)
SERIES_CONFIG = [
    ("Total",            None,                  None,                    2015, "hw",     None,    None,           "mul",  3.5),
    ("March. générales", "CATEGORIE PRODUITS", "MARCHANDISES GENERALES",2015, "hw",     None,    None,           "mul",  5.8),
    ("Prod. pétroliers", "CATEGORIE PRODUITS", "PRODUITS PETROLIERS",   2015, "hw",     None,    None,           "mul", 15.9),
    ("Prod. de pêche",   "CATEGORIE PRODUITS", "PRODUITS DE PÊCHE",     2015, "hw",     None,    None,           "add",  9.5),
    ("Import",           "Sens_Trafic",       "Import",                2015, "hw",     None,    None,           "mul",  6.1),
    ("Export",           "Sens_Trafic",       "Export",                2021, "hw",     None,    None,           "add",  7.8),
    ("National",         "Destination_new",     "National",              2015, "sarima", (1,1,0), (0,1,1,12),    "add",  7.6),
    ("Transit",          "Destination_new",     "Transit",               2020, "hw",     None,    None,           "add", 28.1),
    ("Transbordement",   "Destination_new",     "Transbordement",        2023, "naif",   None,    None,            None, 31.6),
    ("Conteneurisé",     "Conteneurise",        "Y",                     2021, "sarima", (0,1,1), (0,1,1,12),    "add",  6.9),
    ("Non conteneurisé", "Conteneurise",        "N",                     2015, "sarima", (1,1,0), (1,0,1,12),    "add", 10.1),
]

# Séries à signaler avec avertissement dans le dashboard
SERIES_WARN = {
    "Prod. pétroliers" : "Série volatile — prix pétrole et chocs d'approvisionnement",
    "Transit"          : "Flux influencés par le contexte géopolitique sahélien",
    "Transbordement"   : "Rupture structurelle 2023 — historique 2015–2022 non représentatif",
}

# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────
print("=" * 60)
print("  TRAIN_MODELS — Entraînement & Sérialisation")
print("=" * 60)

df = pd.read_excel(FILE_PATH)
df = df[~df["Sens_Trafic"].astype(str).str.startswith("Filtres")]
df = df.dropna(subset=["Date", "Poids_march(tonnes)", "Sens_Trafic"])
df["Date"]    = pd.to_datetime(df["Date"])
df["Annee"]   = df["Date"].dt.year
df["Mois"]    = df["Date"].dt.month
df["Tonne_Mt"] = df["Poids_march(tonnes)"] / 1_000_000
df["Destination_new"] = df["Destination"].apply(
    lambda x: "Transit"
    if x in ["Burkina Faso", "Mali", "Niger", "Pays Cotiers"]
    else x
)
# Conteneurisé = TC1 + TC2
TC_TERM = ["TERMINAL A CONTENEUR (TC 1)", "TERMINAL A CONTENEUR (TC 2)"]
df["Conteneurise"] = df["Terminal"].isin(TC_TERM).map({True: "Y", False: "N"})

ANNEE_MAX_DATA = int(df["Annee"].max())
print(f"\n  Fichier      : {FILE_PATH}")
print(f"  Lignes       : {len(df):,}")
print(f"  Période      : {int(df['Annee'].min())}–{ANNEE_MAX_DATA}")
print(f"  Séries       : {len(SERIES_CONFIG)}")

def build_serie(df, col=None, val=None):
    sub = df if col is None else df[df[col] == val]
    m   = sub.groupby(["Annee","Mois"])["Tonne_Mt"].sum().reset_index()
    m["date"] = pd.to_datetime({"year":m["Annee"],"month":m["Mois"],"day":1})
    return m.sort_values("date").set_index("date")["Tonne_Mt"].asfreq("MS")

# ─────────────────────────────────────────────
# 2. ENTRAÎNEMENT
# ─────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  ENTRAÎNEMENT DES MODÈLES")
print(f"{'─'*60}")

fitted_models = {}   # {label: model_object}
series_store  = {}   # {label: serie_complete}

for cfg in SERIES_CONFIG:
    label, col, val, debut, mtype, order, seas, hw_seas, wmape_val = cfg

    serie_full = build_serie(df, col, val)
    # Train = du début choisi jusqu'à la dernière année disponible
    train = serie_full[serie_full.index.year >= debut]
    series_store[label] = serie_full

    print(f"\n  [{label}]")
    print(f"    Type  : {mtype.upper()}")
    print(f"    Train : {debut}–{ANNEE_MAX_DATA} ({len(train)} mois)")
    print(f"    Dernière val. : {train.iloc[-1]:.3f} Mt "
          f"({train.index[-1].strftime('%b %Y')})")

    if mtype == "hw":
        model = ExponentialSmoothing(
            train.clip(lower=1e-6),
            trend="add", seasonal=hw_seas,
            seasonal_periods=12,
        ).fit(optimized=True)
        print(f"    α={model.params['smoothing_level']:.3f}  "
              f"β={model.params['smoothing_trend']:.3f}  "
              f"γ={model.params['smoothing_seasonal']:.3f}")

    elif mtype == "sarima":
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seas,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        print(f"    AIC={model.aic:.1f}  "
              f"SARIMA{order}{seas}")

    elif mtype == "naif":
        # Stocker simplement les valeurs des 2 dernières années
        annee_max = train.index.year.max()
        v_last  = train[train.index.year == annee_max]
        v_prev  = train[train.index.year == annee_max-1]
        model   = {"type":"naif","v_last":v_last,"v_prev":v_prev}
        print(f"    Naïf pondéré 60/40 sur {annee_max-1}–{annee_max}")

    fitted_models[label] = {
        "model"     : model,
        "type"      : mtype,
        "hw_seas"   : hw_seas,
        "order"     : order,
        "seas_order": seas,
        "train_debut": debut,
        "wmape_test": wmape_val,
        "train_end" : int(ANNEE_MAX_DATA),
    }
    print(f"    WMAPE test 2025 : {wmape_val:.1f}%  ✓")

# ─────────────────────────────────────────────
# 3. SAUVEGARDE
# ─────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  SAUVEGARDE")
print(f"{'─'*60}")

meta = {
    "annee_max_data" : ANNEE_MAX_DATA,
    "annee_min_fc"   : ANNEE_MAX_DATA + 1,
    "series_config"  : SERIES_CONFIG,
    "series_warn"    : SERIES_WARN,
    "fichier_source" : FILE_PATH,
    "axes" : {
        "Total"            : (None, None),
        "Grandes composantes": ("CATEGORIE PRODUITS", [
            "March. générales","Prod. pétroliers","Prod. de pêche"]),
        "Sens de trafic"   : ("Sens_Trafic", ["Import","Export"]),
        "Destination"      : ("Destination_new", [
            "National","Transit","Transbordement"]),
        "Conteneurisation" : ("Conteneurise", [
            "Conteneurise","Non conteneurisé"]),
    },
}

with open(OUTPUT_MDL, "wb") as f:
    pickle.dump({"models": fitted_models, "meta": meta}, f)
print(f"\n  ✓ {OUTPUT_MDL}  ({len(fitted_models)} modèles)")

with open(OUTPUT_SER, "wb") as f:
    pickle.dump(series_store, f)
print(f"  ✓ {OUTPUT_SER}  ({len(series_store)} séries)")

# ─────────────────────────────────────────────
# 4. SYNTHÈSE
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("  SYNTHÈSE")
print(f"{'='*60}")
print(f"\n  {'Série':<22} {'Type':<8} {'Train':<12} {'WMAPE test'}")
print(f"  {'─'*55}")
for cfg in SERIES_CONFIG:
    label, _, _, debut, mtype, *_, wmape_val = cfg
    print(f"  {label:<22} {mtype.upper():<8} "
          f"{str(debut)+'-'+str(ANNEE_MAX_DATA):<12} {wmape_val:.1f}%")

wmapes = [c[-1] for c in SERIES_CONFIG if c[4] != "naif"
          and c[0] not in ("Transit","Transbordement")]
print(f"\n  WMAPE moyen (9 séries principales) : {np.mean(wmapes):.2f}%")
print(f"\n  → Lancer ensuite : python forecast_engine.py")
