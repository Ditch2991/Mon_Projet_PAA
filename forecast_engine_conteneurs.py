"""
forecast_engine_conteneurs.py — Moteur de prévision conteneurs Port d'Abidjan
==============================================================================
Étape 1 : Prévision Total annuel (Holt amortie, train 2023-2025)
Étape 2 : Ventilation mensuelle (profil saisonnier 2023-2025)
Étape 3 : Répartition par terminal (Top-down clés N-1)
Étape 4 : Répartition par destination (Top-down clés N-1)
          dont Transbordé = TC2 (constant) + Habituel (Holt)

Usage    : python forecast_engine_conteneurs.py
Prérequis: train_models_conteneurs.py doit avoir été exécuté
Sortie   : forecasts_conteneurs.pkl
"""

import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────────
INPUT_MDL = "models_conteneurs.pkl"
INPUT_SER = "series_conteneurs.pkl"
OUTPUT_FC = "forecasts_conteneurs.pkl"
ANNEE_MAX = 2040

# ─────────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("FORECAST_ENGINE_CONTENEURS — Port d'Abidjan")
print("=" * 65)

with open(INPUT_MDL, 'rb') as f: mdl = pickle.load(f)
with open(INPUT_SER, 'rb') as f: series = pickle.load(f)

ANNEE_FIN   = mdl['annee_fin']
ANNEE_DEBUT = mdl['annee_debut']
SEGS_TERM   = mdl['segs_term']
SEGS_DEST   = mdl['segs_dest']
profil      = mdl['profil_saisonnier']
parts_term  = mdl['parts_term']
parts_dest  = mdl['parts_dest']
noms_m      = mdl['noms_mois']
H           = ANNEE_MAX - ANNEE_FIN

print(f"\n[1] Modèles chargés")
print(f"    Total     : α={mdl['alpha_tot']:.2f} β={mdl['beta_tot']:.2f} "
      f"φ={mdl['phi_tot']:.2f}  err={mdl['err_tot']:.1f}%")
print(f"    Non transb: α={mdl['params_nt'][0]:.2f} β={mdl['params_nt'][1]:.2f} "
      f"φ={mdl['params_nt'][2]:.2f}  WMAPE={mdl['wmape_nt']:.1f}%")
print(f"    Transb TC2: constant = {mdl['transb_tc2_2025']:,} TEU/an")
print(f"    Transb hab: α={mdl['params_th'][0]:.2f} β={mdl['params_th'][1]:.2f} "
      f"φ={mdl['params_th'][2]:.2f}  WMAPE={mdl['wmape_th']:.1f}%")

# ─────────────────────────────────────────────────────────────────
# 2. FONCTION HOLT AMORTIE
# ─────────────────────────────────────────────────────────────────
def holt_damped(y, alpha, beta, phi, h=1):
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

# ─────────────────────────────────────────────────────────────────
# 3. PRÉVISIONS TOTAL ANNUEL
# ─────────────────────────────────────────────────────────────────
print(f"\n[2] Prévisions Total annuel {ANNEE_FIN+1}-{ANNEE_MAX} ...")

fc_tot_raw, _, _, _ = holt_damped(
    mdl['y_tot_court'], mdl['alpha_tot'], mdl['beta_tot'], mdl['phi_tot'], h=H)
fc_tot  = np.round(fc_tot_raw).astype(int)
rmse_t  = mdl['rmse_tot']
ic_lo_t = np.round(fc_tot_raw - 1.96 * rmse_t * np.sqrt(np.arange(1, H+1))).astype(int)
ic_hi_t = np.round(fc_tot_raw + 1.96 * rmse_t * np.sqrt(np.arange(1, H+1))).astype(int)

for i, yr in enumerate(range(ANNEE_FIN+1, ANNEE_MAX+1)):
    print(f"    {yr} : {fc_tot[i]:>12,}  [IC95%: {ic_lo_t[i]:>12,} – {ic_hi_t[i]:>12,}]")

# ─────────────────────────────────────────────────────────────────
# 4. PRÉVISIONS NON TRANSBORDÉ
# ─────────────────────────────────────────────────────────────────
fc_nt_raw, _, _, _ = holt_damped(
    mdl['y_nt'], *mdl['params_nt'], h=H)
fc_nt = np.round(fc_nt_raw).astype(int)

# ─────────────────────────────────────────────────────────────────
# 5. PRÉVISIONS TRANSBORDÉ TC2 (constant)
# ─────────────────────────────────────────────────────────────────
fc_tc2 = np.array([mdl['transb_tc2_2025']] * H)

# ─────────────────────────────────────────────────────────────────
# 6. TRANSBORDÉ HABITUEL = RÉSIDUEL (pas de modèle)
# Calculé comme : Total - (Non transb. + Transb. TC2) → voir boucle
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
# 7. PRÉVISIONS COMPLÈTES : mensuel + top-down récursif
# ─────────────────────────────────────────────────────────────────
print(f"\n[3] Ventilation mensuelle + Top-down récursif ...")

# Clés initiales = parts réelles 2025
parts_td_term = {ANNEE_FIN: parts_term[ANNEE_FIN]}
parts_td_dest = {ANNEE_FIN: parts_dest[ANNEE_FIN]}

def ann_hist(seg, yr):
    s = series[seg]
    return int(s[s.index.year == yr].sum())

forecasts = {}

# Stocker historique
for yr in range(ANNEE_DEBUT, ANNEE_FIN + 1):
    tot_yr   = mdl['ann_total'][yr]
    mens_yr  = series['Total'][series['Total'].index.year == yr].values.astype(int)
    segs_t_yr = {g: series[g][series[g].index.year == yr].values.astype(int)
                 for g in SEGS_TERM}
    segs_d_yr = {
        'Non transb.'     : series['Non transb.'][series['Non transb.'].index.year == yr].values.astype(int),
        'Transbordé'      : series['Transbordé'][series['Transbordé'].index.year == yr].values.astype(int),
        'Transb. TC2'     : series['Transb. TC2'][series['Transb. TC2'].index.year == yr].values.astype(int),
        'Transb. habituel': series['Transb. habituel'][series['Transb. habituel'].index.year == yr].values.astype(int),
    }
    forecasts[('historique', yr)] = {
        'annuel'          : tot_yr,
        'mensuel'         : mens_yr,
        'segments_term'   : segs_t_yr,
        'segments_dest'   : segs_d_yr,
    }

# Prévisions 2026-2040
for i, yr in enumerate(range(ANNEE_FIN+1, ANNEE_MAX+1)):
    tot_ann = int(fc_tot[i])
    lo_ann  = int(ic_lo_t[i])
    hi_ann  = int(ic_hi_t[i])

    # ── Ventilation mensuelle ──
    mens = np.round(tot_ann * profil / 100).astype(int)
    diff = tot_ann - mens.sum()
    if diff != 0:
        mens[np.argmax(profil)] += diff

    mens_lo = np.round(lo_ann * profil / 100).astype(int)
    mens_hi = np.round(hi_ann * profil / 100).astype(int)

    # ── Top-down terminal (clés N-1) ──
    cle_t = parts_td_term[yr - 1]
    segs_term = {}
    for g in SEGS_TERM:
        sv = np.round(mens * cle_t[g] / 100).astype(int)
        segs_term[g] = sv

    # Ajustement arrondi terminal : forcer somme segments = total mensuel
    seg_dom_t = SEGS_TERM[1]  # TC2 = segment dominant en 2025+
    for m in range(12):
        ecart_m = mens[m] - sum(segs_term[g][m] for g in SEGS_TERM)
        if ecart_m != 0:
            segs_term[seg_dom_t][m] += ecart_m

    # Mise à jour clés terminaux
    parts_td_term[yr] = {
        g: segs_term[g].sum() / tot_ann * 100 for g in SEGS_TERM
    }

    # ── Top-down destination (clés N-1) ──
    cle_d = parts_td_dest[yr - 1]
    segs_dest = {}

    # Non transbordé et Transbordé total par clés
    nt_mens      = np.round(mens * cle_d['Non transb.'] / 100).astype(int)
    transb_mens  = mens - nt_mens   # garantit la somme = total

    # Transbordé TC2 (constant annuel → ventilé par profil)
    tc2_ann  = int(fc_tc2[i])
    tc2_mens = np.round(tc2_ann * profil / 100).astype(int)
    diff_tc2 = tc2_ann - tc2_mens.sum()
    if diff_tc2 != 0: tc2_mens[np.argmax(profil)] += diff_tc2

    # Transbordé habituel = Transbordé total - TC2
    th_mens = transb_mens - tc2_mens
    th_mens = np.maximum(th_mens, 0)   # pas de négatifs

    segs_dest['Non transb.']      = nt_mens
    segs_dest['Transbordé']       = transb_mens
    segs_dest['Transb. TC2']      = tc2_mens
    segs_dest['Transb. habituel'] = th_mens

    # Mise à jour clés destination
    parts_td_dest[yr] = {
        'Non transb.'     : nt_mens.sum()      / tot_ann * 100,
        'Transbordé'      : transb_mens.sum()  / tot_ann * 100,
        'Transb. TC2'     : tc2_mens.sum()     / tot_ann * 100,
        'Transb. habituel': th_mens.sum()      / tot_ann * 100,
    }

    forecasts[yr] = {
        'annuel'          : tot_ann,
        'ic_lo'           : lo_ann,
        'ic_hi'           : hi_ann,
        'mensuel'         : mens,
        'mensuel_lo'      : mens_lo,
        'mensuel_hi'      : mens_hi,
        'segments_term'   : segs_term,
        'segments_dest'   : segs_dest,
        'parts_td_term'   : {g: round(parts_td_term[yr][g], 4) for g in SEGS_TERM},
        'parts_td_dest'   : {k: round(parts_td_dest[yr][k], 4) for k in parts_td_dest[yr]},
        'cle_annee'       : yr - 1,
        'transb_tc2_ann'  : tc2_ann,
        'transb_hab_ann'  : int(th_mens.sum()),  # Résiduel : Total - (NT + TC2)
    }

    # Vérification
    check_t = sum(segs_term[g].sum() for g in SEGS_TERM)
    check_d = segs_dest['Non transb.'].sum() + segs_dest['Transbordé'].sum()
    print(f"    {yr} : Total={tot_ann:>10,}  "
          f"ΣTerminaux={check_t:>10,} (écart={check_t-tot_ann:+d})  "
          f"ΣDest={check_d:>10,} (écart={check_d-tot_ann:+d})")

# ─────────────────────────────────────────────────────────────────
# 8. MÉTADONNÉES + SAUVEGARDE
# ─────────────────────────────────────────────────────────────────
meta = {
    'modele_total'     : 'Holt amortie',
    'alpha_tot'        : mdl['alpha_tot'],
    'beta_tot'         : mdl['beta_tot'],
    'phi_tot'          : mdl['phi_tot'],
    'err_tot'          : mdl['err_tot'],
    'rmse_tot'         : mdl['rmse_tot'],
    'wmape_nt'         : mdl['wmape_nt'],
    'methode_transb_hab': 'Résiduel : Total - (Non transb. + Transb. TC2)',
    'transb_tc2_2025'  : mdl['transb_tc2_2025'],
    'profil_saisonnier': profil,
    'annees_profil'    : mdl['annees_profil'],
    'parts_term'       : parts_term,
    'parts_dest'       : parts_dest,
    'parts_td_term'    : parts_td_term,
    'parts_td_dest'    : parts_td_dest,
    'segs_term'        : SEGS_TERM,
    'segs_dest'        : SEGS_DEST,
    'annee_debut'      : ANNEE_DEBUT,
    'annee_fin'        : ANNEE_FIN,
    'annee_rupture'    : mdl['annee_rupture'],
    'annee_max'        : ANNEE_MAX,
    'noms_mois'        : noms_m,
    'ann_total_hist'   : mdl['ann_total'],
}

output = {'forecasts': forecasts, 'meta': meta}

with open(OUTPUT_FC, 'wb') as f:
    pickle.dump(output, f)

print(f"\n[4] Fichier sauvegardé : {OUTPUT_FC}")
print(f"\n{'='*65}")
print(f"✓ Prévisions générées — {ANNEE_FIN+1} à {ANNEE_MAX} ({H} ans)")
print(f"  Total     : Holt amortie (err={mdl['err_tot']:.1f}%)  train {mdl['annee_rupture']}-{ANNEE_FIN}")
print(f"  Non transb: Holt amortie (WMAPE={mdl['wmape_nt']:.1f}%)")
print(f"  Transb TC2: constant {mdl['transb_tc2_2025']:,} TEU/an")
print(f"  Transb hab: R\u00e9siduel Total - (Non transb. + Transb. TC2)")
print(f"  Terminaux : Top-down clés N-1 récursif")
print(f"{'='*65}")
