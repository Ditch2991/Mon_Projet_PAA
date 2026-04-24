"""
forecast_engine_escales.py — Moteur de prévision escales Port d'Abidjan
========================================================================
Étape 1 : Prévision Total annuel (Holt amortie)
Étape 2 : Ventilation mensuelle (profil saisonnier 2023-2025)
Étape 3 : Répartition par terminal (Top-down, clés N-1 récursif)

Usage    : python forecast_engine_escales.py
Prérequis: train_models_escales.py doit avoir été exécuté
Sortie   : forecasts_escales.pkl
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────────
INPUT_MDL   = "models_escales.pkl"
INPUT_SER   = "series_escales.pkl"
OUTPUT_FC   = "forecasts_escales.pkl"
ANNEE_MAX   = 2040

# ─────────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("FORECAST_ENGINE_ESCALES — Port d'Abidjan")
print("=" * 65)

with open(INPUT_MDL, 'rb') as f:
    mdl = pickle.load(f)

with open(INPUT_SER, 'rb') as f:
    series_mens = pickle.load(f)

alpha   = mdl['alpha']
beta    = mdl['beta']
phi     = mdl['phi']
wmape   = mdl['wmape']
rmse    = mdl['rmse']
L_fin   = mdl['L_final']
T_fin   = mdl['T_final']
y_train = mdl['y_train']
profil  = mdl['profil_saisonnier']
parts_h = mdl['parts_terminaux']   # historique des clés
SEGS    = mdl['segs']
ANNEE_FIN = mdl['annee_fin']

print(f"\n[1] Modèle chargé : Holt amortie α={alpha:.2f} β={beta:.2f} φ={phi:.2f}")
print(f"    WMAPE={wmape:.1f}%  RMSE={rmse:.1f} escales")

# ─────────────────────────────────────────────────────────────────
# 2. PRÉVISIONS TOTAL ANNUEL (Holt amortie)
# ─────────────────────────────────────────────────────────────────
H = ANNEE_MAX - ANNEE_FIN   # nombre d'années à prévoir

def holt_damped_forecast(y, alpha, beta, phi, h=1):
    n = len(y)
    L = np.zeros(n); T = np.zeros(n)
    L[0] = y[0]; T[0] = y[1] - y[0]
    for t in range(1, n):
        L[t] = alpha * y[t] + (1 - alpha) * (L[t-1] + phi * T[t-1])
        T[t] = beta * (L[t] - L[t-1]) + (1 - beta) * phi * T[t-1]
    fc = np.array([L[-1] + sum(phi**j * T[-1] for j in range(1, i+1))
                   for i in range(1, h + 1)])
    fitted = np.array([L[t-1] + phi * T[t-1] for t in range(1, n)])
    return fc, fitted, L, T

fc_raw, _, _, _ = holt_damped_forecast(y_train, alpha, beta, phi, h=H)
fc_ann  = np.round(fc_raw).astype(int)
ic_lo   = np.round(fc_raw - 1.96 * rmse * np.sqrt(np.arange(1, H+1))).astype(int)
ic_hi   = np.round(fc_raw + 1.96 * rmse * np.sqrt(np.arange(1, H+1))).astype(int)

print(f"\n[2] Prévisions Total annuel 2026-{ANNEE_MAX} :")
for i, yr in enumerate(range(ANNEE_FIN + 1, ANNEE_MAX + 1)):
    print(f"    {yr} : {fc_ann[i]:>5}  [IC95%: {ic_lo[i]:>5} – {ic_hi[i]:>5}]")

# ─────────────────────────────────────────────────────────────────
# 3. PRÉVISIONS COMPLÈTES : mensuel + top-down récursif
# ─────────────────────────────────────────────────────────────────
print(f"\n[3] Ventilation mensuelle + Top-down récursif ...")

noms_m = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']

# Initialiser les clés top-down avec les parts réelles de la dernière année
parts_td = {ANNEE_FIN: parts_h[ANNEE_FIN]}

forecasts = {}

# Stocker aussi les données historiques dans forecasts pour le dashboard
for yr in range(mdl['annee_debut'], ANNEE_FIN + 1):
    tot_yr = int(series_mens['Total'][series_mens['Total'].index.year == yr].sum())
    mens_yr = series_mens['Total'][series_mens['Total'].index.year == yr].values.astype(int)
    segs_yr = {g: series_mens[g][series_mens[g].index.year == yr].values.astype(int)
               for g in SEGS}
    forecasts[('historique', yr)] = {
        'annuel'  : tot_yr,
        'mensuel' : mens_yr,
        'segments': segs_yr,
    }

# Prévisions 2026-2040
for i, yr in enumerate(range(ANNEE_FIN + 1, ANNEE_MAX + 1)):
    tot_ann = int(fc_ann[i])
    lo_ann  = int(ic_lo[i])
    hi_ann  = int(ic_hi[i])

    # ── Étape 2 : ventilation mensuelle par profil saisonnier ──
    mens = np.round(tot_ann * profil / 100).astype(int)
    # Ajustement arrondi pour que la somme = total exact
    diff = tot_ann - mens.sum()
    if diff != 0:
        mens[np.argmax(profil)] += diff

    mens_lo = np.round(lo_ann * profil / 100).astype(int)
    mens_hi = np.round(hi_ann * profil / 100).astype(int)

    # ── Étape 3 : répartition par terminal (clés N-1) ──
    cle = parts_td[yr - 1]
    segs_mens = {}
    for g in SEGS:
        sv = np.round(mens * cle[g] / 100).astype(int)
        segs_mens[g] = sv

    # Ajustement arrondi : forcer la somme des segments = total mensuel exact
    # On ajuste mois par mois sur le segment dominant (TC1)
    seg_dom = SEGS[0]  # TC1 = segment dominant
    for m in range(12):
        ecart_m = mens[m] - sum(segs_mens[g][m] for g in SEGS)
        if ecart_m != 0:
            segs_mens[seg_dom][m] += ecart_m

    # Mise à jour des clés pour l'année suivante (parts top-down de cette année)
    parts_td[yr] = {
        g: segs_mens[g].sum() / tot_ann * 100 for g in SEGS
    }

    forecasts[yr] = {
        'annuel'    : tot_ann,
        'ic_lo'     : lo_ann,
        'ic_hi'     : hi_ann,
        'mensuel'   : mens,
        'mensuel_lo': mens_lo,
        'mensuel_hi': mens_hi,
        'segments'  : segs_mens,
        'parts_td'  : {g: round(parts_td[yr][g], 4) for g in SEGS},
        'cle_annee' : yr - 1,
    }

    # Vérif cohérence
    check = sum(segs_mens[g].sum() for g in SEGS)
    ecart = check - tot_ann
    print(f"    {yr} : Total={tot_ann:>5}  Somme segments={check:>5}  écart={ecart:+d}")

# ─────────────────────────────────────────────────────────────────
# 4. MÉTADONNÉES GLOBALES
# ─────────────────────────────────────────────────────────────────
meta = {
    'modele'           : 'Holt amortie',
    'alpha'            : alpha,
    'beta'             : beta,
    'phi'              : phi,
    'wmape'            : wmape,
    'rmse'             : rmse,
    'profil_saisonnier': profil,
    'annees_profil'    : mdl['annees_profil'],
    'parts_terminaux'  : parts_h,           # parts historiques réelles
    'parts_td'         : parts_td,          # parts top-down calculées
    'segs'             : SEGS,
    'annee_debut'      : mdl['annee_debut'],
    'annee_fin'        : ANNEE_FIN,
    'annee_max'        : ANNEE_MAX,
    'noms_mois'        : noms_m,
    'ann_total_hist'   : mdl['ann_total'],
}

# ─────────────────────────────────────────────────────────────────
# 5. SAUVEGARDE
# ─────────────────────────────────────────────────────────────────
output = {
    'forecasts': forecasts,
    'meta'     : meta,
}

with open(OUTPUT_FC, 'wb') as f:
    pickle.dump(output, f)

print(f"\n[4] Fichier sauvegardé : {OUTPUT_FC}")
print(f"\n{'='*65}")
print(f"✓ Prévisions générées — {ANNEE_FIN+1} à {ANNEE_MAX} ({H} ans)")
print(f"  Modèle  : Holt amortie (WMAPE={wmape:.1f}%)")
print(f"  Profil  : moyenne {mdl['annees_profil']}")
print(f"  Top-down: clés N-1 récursif")
print(f"{'='*65}")
