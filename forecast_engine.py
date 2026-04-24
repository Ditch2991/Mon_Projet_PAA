"""
forecast_engine.py — Moteur de prévision récursif + Réconciliation
===================================================================
Tout-en-un : prévisions brutes + Top-down + Bottom-up intégrés.

Usage    : python forecast_engine.py
Prérequis: train_models.py doit avoir été exécuté (models.pkl + series.pkl)
Sortie   : forecasts.pkl
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
ANNEE_MAX_FC = 2040
INPUT_MDL    = "models.pkl"
INPUT_SER    = "series.pkl"
OUTPUT_FC    = "forecasts.pkl"

# Approche par défaut pour le dashboard et les exports Excel
DEFAULT_APPROCHE = "top_down"

# ─────────────────────────────────────────────
# 1. STRUCTURE DES AXES (réconciliation)
# ─────────────────────────────────────────────
AXES = {
    "sens": {
        "segments": ["Import", "Export"],
    },
    "composante": {
        "segments": ["March. générales", "Prod. pétroliers", "Prod. de pêche"],
    },
    "destination": {
        "segments": ["National", "Transit", "Transbordement"],
    },
    "conteneur": {
        "segments": ["Conteneurisé", "Non conteneurisé"],
    },
}

# ─────────────────────────────────────────────
# 2. CHARGEMENT
# ─────────────────────────────────────────────
print("=" * 60)
print("  FORECAST_ENGINE — Prévisions récursives + Réconciliation")
print("=" * 60)

with open(INPUT_MDL, "rb") as f:
    mdl_data = pickle.load(f)
with open(INPUT_SER, "rb") as f:
    series_store = pickle.load(f)

fitted_models = mdl_data["models"]
meta          = mdl_data["meta"]
ANNEE_MIN_FC  = meta["annee_min_fc"]
ANNEE_MAX_DATA= meta["annee_max_data"]

print(f"\n  Modèles chargés  : {len(fitted_models)}")
print(f"  Données jusqu'à  : {ANNEE_MAX_DATA}")
print(f"  Prévisions       : {ANNEE_MIN_FC}–{ANNEE_MAX_FC}")

# ─────────────────────────────────────────────
# 3. FONCTIONS MODÈLES
# ─────────────────────────────────────────────

def refit_hw(serie_aug, hw_seas):
    return ExponentialSmoothing(
        serie_aug.clip(lower=1e-6),
        trend="add", seasonal=hw_seas,
        seasonal_periods=12,
    ).fit(optimized=True)


def refit_sarima(serie_aug, order, seas_order):
    return SARIMAX(
        serie_aug,
        order=order,
        seasonal_order=seas_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)


def naif_forecast(serie_aug, n=12):
    annee_max = serie_aug.index.year.max()
    preds = []
    for m in range(1, n + 1):
        v_last = serie_aug[
            (serie_aug.index.year == annee_max) &
            (serie_aug.index.month == m)
        ]
        v_prev = serie_aug[
            (serie_aug.index.year == annee_max - 1) &
            (serie_aug.index.month == m)
        ]
        if len(v_last) > 0 and len(v_prev) > 0:
            preds.append(0.6 * v_last.values[0] + 0.4 * v_prev.values[0])
        elif len(v_last) > 0:
            preds.append(v_last.values[0])
        else:
            preds.append(float(serie_aug.mean()))
    return np.array(preds)


def get_ic(model_obj, mtype, fc_vals, serie_aug):
    if mtype == "sarima":
        try:
            ci    = model_obj.get_forecast(12).conf_int(alpha=0.05)
            ic_lo = ci.iloc[:, 0].clip(lower=0).values
            ic_hi = ci.iloc[:, 1].values
            return ic_lo, ic_hi
        except Exception:
            pass
    if mtype == "hw":
        try:
            resid_std = float((serie_aug - model_obj.fittedvalues).std())
        except Exception:
            resid_std = float(serie_aug.std() * 0.10)
    else:
        resid_std = float(serie_aug.std() * 0.15)
    ic_lo = np.maximum(fc_vals - 1.96 * resid_std, 0)
    ic_hi = fc_vals + 1.96 * resid_std
    return ic_lo, ic_hi

# ─────────────────────────────────────────────
# 4. FONCTIONS RÉCONCILIATION
# ─────────────────────────────────────────────

def get_parts_annee(series_store, annee):
    """Parts de chaque segment dans le total pour une année donnée."""
    total_serie = series_store.get("Total")
    if total_serie is None:
        return {}, 0
    total_ann = total_serie[total_serie.index.year == annee].sum()
    if total_ann == 0:
        return {}, 0
    parts = {}
    for axe, cfg in AXES.items():
        parts[axe] = {}
        segments = cfg["segments"]
        for seg in segments:
            serie = series_store.get(seg)
            if serie is None:
                parts[axe][seg] = 1.0 / len(segments)
                continue
            val = serie[serie.index.year == annee].sum()
            parts[axe][seg] = val / total_ann if total_ann > 0 else 0
    return parts, float(total_ann)


def reconcile_top_down(fc_total_mensuel, parts_axe):
    """Distribue le total mensuel selon les parts de l'année précédente."""
    return {
        seg: np.array([v * part for v in fc_total_mensuel])
        for seg, part in parts_axe.items()
    }


def reconcile_bottom_up(fc_segments_mensuels):
    """Recalcule le total comme somme des segments bruts."""
    arrays = list(fc_segments_mensuels.values())
    if not arrays:
        return np.zeros(12)
    return np.sum(arrays, axis=0)


def apply_reconciliations(all_forecasts, series_store,
                           annee_min_fc, annee_max_fc):
    """
    Applique Top-down et Bottom-up pour toutes les années et tous les axes.
    Clé Top-down année N = parts réalisées/prévues de N-1.
    """
    print("\n[RÉCONCILIATION] Top-down & Bottom-up...")

    # Parts initiales = dernière année de données réelles
    parts_td, _ = get_parts_annee(series_store, annee_min_fc - 1)

    for yr in range(annee_min_fc, annee_max_fc + 1):

        fc_total = all_forecasts.get(("Total", yr))
        if fc_total is None:
            continue
        total_mensuel = fc_total["fc"]

        for axe, cfg in AXES.items():
            segments  = cfg["segments"]
            parts_axe = parts_td.get(axe, {})

            # Prévisions brutes des segments
            fc_segs_bruts = {
                seg: all_forecasts.get((seg, yr), {}).get("fc", np.zeros(12))
                for seg in segments
            }

            # ── Top-down ──
            td_result = reconcile_top_down(total_mensuel, parts_axe)
            for seg in segments:
                key = (seg, yr)
                if key not in all_forecasts:
                    all_forecasts[key] = {"fc": np.zeros(12)}
                all_forecasts[key].setdefault("top_down", {})
                all_forecasts[key]["top_down"][axe] = td_result.get(
                    seg, np.zeros(12))
                all_forecasts[key].setdefault("annuel_td", {})
                all_forecasts[key]["annuel_td"][axe] = float(
                    td_result.get(seg, np.zeros(12)).sum())

            # ── Bottom-up ──
            bu_total = reconcile_bottom_up(fc_segs_bruts)
            all_forecasts[("Total", yr)].setdefault("bottom_up", {})
            all_forecasts[("Total", yr)]["bottom_up"][axe] = bu_total
            all_forecasts[("Total", yr)].setdefault("annuel_bu", {})
            all_forecasts[("Total", yr)]["annuel_bu"][axe] = float(
                bu_total.sum())

        # Mise à jour des parts pour l'année suivante
        # (basées sur les prévisions Top-down de cette année)
        new_parts = {}
        total_td_ann = float(total_mensuel.sum())
        for axe, cfg in AXES.items():
            new_parts[axe] = {}
            for seg in cfg["segments"]:
                val = float(all_forecasts.get((seg, yr), {})
                            .get("top_down", {})
                            .get(axe, np.zeros(12)).sum())
                new_parts[axe][seg] = (val / total_td_ann
                                       if total_td_ann > 0 else 0)
        parts_td = new_parts

        print(f"    {yr} ✓  (clé = parts {yr - 1})")

    print(f"  Terminé — {annee_max_fc - annee_min_fc + 1} années réconciliées")
    return all_forecasts


def verifier_coherence(all_forecasts, annee):
    """Vérifie la cohérence des deux approches pour une année."""
    print(f"\n=== COHÉRENCE {annee} ===")
    fc_total = all_forecasts.get(("Total", annee))
    if fc_total is None:
        print("  Total non disponible"); return

    total_modele = float(fc_total["fc"].sum())
    print(f"  Total modèle HW    : {total_modele:.3f} Mt")
    print()

    print("  TOP-DOWN (Total × parts N-1) :")
    for axe, cfg in AXES.items():
        somme = sum(
            float(all_forecasts.get((seg, annee), {})
                  .get("top_down", {})
                  .get(axe, np.zeros(12)).sum())
            for seg in cfg["segments"]
        )
        ecart = somme - total_modele
        ok    = abs(ecart) < 0.01
        print(f"    {axe:<15} somme={somme:.3f} Mt  "
              f"écart={ecart:+.4f}  {'✓' if ok else '✗'}")

    print()
    print("  BOTTOM-UP (somme des modèles bruts) :")
    for axe in AXES:
        bu = float(fc_total.get("bottom_up", {})
                   .get(axe, np.zeros(12)).sum())
        ecart = bu - total_modele
        print(f"    {axe:<15} total BU={bu:.3f} Mt  "
              f"écart vs modèle={ecart:+.3f} Mt")

# ─────────────────────────────────────────────
# 5. PRÉVISIONS RÉCURSIVES (modèles bruts)
# ─────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  CALCUL DES PRÉVISIONS RÉCURSIVES")
print(f"{'─'*60}")

all_forecasts = {}

for label, mdl_info in fitted_models.items():
    mtype      = mdl_info["type"]
    hw_seas    = mdl_info["hw_seas"]
    order      = mdl_info["order"]
    seas_order = mdl_info["seas_order"]
    debut      = mdl_info["train_debut"]

    serie_full = series_store[label]
    print(f"\n  [{label}]  {mtype.upper()}", end="")

    # Série augmentée — historique réel complet depuis train_debut
    serie_aug = serie_full[serie_full.index.year >= debut].copy()

    for annee_cible in range(ANNEE_MIN_FC, ANNEE_MAX_FC + 1):

        try:
            if mtype == "hw":
                model_obj = refit_hw(serie_aug, hw_seas)
                fc_vals   = model_obj.forecast(12).clip(lower=0).values

            elif mtype == "sarima":
                model_obj = refit_sarima(serie_aug, order, seas_order)
                fc_vals   = (model_obj.get_forecast(12)
                             .predicted_mean.clip(lower=0).values)

            elif mtype == "naif":
                model_obj = None
                fc_vals   = naif_forecast(serie_aug, 12)

            else:
                continue

            ic_lo, ic_hi = get_ic(model_obj, mtype, fc_vals, serie_aug)

            fc_dates = pd.date_range(
                f"{annee_cible}-01-01", periods=12, freq="MS"
            )

            all_forecasts[(label, annee_cible)] = {
                "fc"      : fc_vals,
                "ic_lo"   : ic_lo,
                "ic_hi"   : ic_hi,
                "dates"   : fc_dates,
                "annuel"  : round(float(fc_vals.sum()), 3),
                "mensuel" : {m + 1: round(float(v), 4)
                             for m, v in enumerate(fc_vals)},
            }

            # Enrichir l'historique pour l'année suivante
            new_pts   = pd.Series(fc_vals, index=fc_dates)
            serie_aug = pd.concat([serie_aug, new_pts])
            serie_aug = serie_aug[~serie_aug.index.duplicated(keep="last")]
            serie_aug = serie_aug.asfreq("MS")

        except Exception as e:
            print(f"\n    ERREUR {annee_cible}: {e}")
            continue

    fc_2026 = all_forecasts.get((label, ANNEE_MIN_FC), {}).get("annuel", "?")
    fc_end  = all_forecasts.get((label, ANNEE_MAX_FC),  {}).get("annuel", "?")
    print(f"  {ANNEE_MIN_FC}–{ANNEE_MAX_FC} ✓  |  "
          f"{ANNEE_MIN_FC}={fc_2026} Mt  {ANNEE_MAX_FC}={fc_end} Mt")

# ─────────────────────────────────────────────
# 6. RÉCONCILIATION
# ─────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  RÉCONCILIATION")
print(f"{'─'*60}")

all_forecasts = apply_reconciliations(
    all_forecasts, series_store, ANNEE_MIN_FC, ANNEE_MAX_FC
)
verifier_coherence(all_forecasts, ANNEE_MIN_FC)

# ─────────────────────────────────────────────
# 7. SAUVEGARDE
# ─────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  SAUVEGARDE")
print(f"{'─'*60}")

payload = {
    "forecasts"       : all_forecasts,
    "series_store"    : series_store,
    "meta"            : meta,
    "annee_max_fc"    : ANNEE_MAX_FC,
    "annee_min_fc"    : ANNEE_MIN_FC,
    "annee_max_data"  : ANNEE_MAX_DATA,
    "default_approche": DEFAULT_APPROCHE,
}

with open(OUTPUT_FC, "wb") as f:
    pickle.dump(payload, f)

nb = len(all_forecasts)
print(f"\n  ✓ {OUTPUT_FC}")
print(f"    {nb} entrées")
print(f"    Réconciliation Top-down & Bottom-up intégrée")
print(f"    Approche par défaut : {DEFAULT_APPROCHE}")
print(f"\n  → Lancer ensuite : streamlit run dashboard.py")
