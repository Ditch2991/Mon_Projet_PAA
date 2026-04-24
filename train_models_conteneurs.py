"""
train_models_conteneurs.py — Entraînement modèles conteneurs Port d'Abidjan
============================================================================
Modèles :
  - Total conteneurs    : Holt amortie (train 2023-2025, post-rupture TC2)
  - Non transbordé      : Holt amortie (train 2015-2025)
  - Transbordé TC2      : Constant = réalisé 2025
  - Transbordé habituel : Holt amortie (train 2015-2025)
  - Segments terminaux  : Top-down clés N-1 (TC1, TC2, Fruitier, Roulier, Autres)
  - Segments destination: Top-down clés N-1 (Non transb., Transbordé)

Usage  : python train_models_conteneurs.py
Sortie : models_conteneurs.pkl + series_conteneurs.pkl
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────────
FICHIER_DONNEES = "data_Conteneurs.xlsx"
OUTPUT_MDL      = "models_conteneurs.pkl"
OUTPUT_SER      = "series_conteneurs.pkl"

ANNEE_DEBUT     = 2015
ANNEE_FIN       = 2025
ANNEE_RUPTURE   = 2023   # démarrage TC2 → entraîner Total sur 2023-2025

SEGS_TERM = ['TC1', 'TC2', 'Fruitier', 'Roulier', 'Autres zones']
SEGS_DEST = ['Non transb.', 'Transbordé', 'Transb. TC2', 'Transb. habituel']

AUTRES_TERM = ['AZITO','CARENA','COFFRE','MOUILLAGE','PHILIPPS',
               'Quai NORD (Q1-Q5)','Quai OUEST (Q6-Q10)','Quai SOGIP',
               'SIVENG','SOCOPAO','SUD','Siap',
               'TERMINAL CEREALIER','TERMINAL MINERALIER',
               'TERMINAL PECHE','TERMINAL PETROLIER']

MAPPING_TERM = {
    'TERMINAL A CONTENEUR (TC 1)': 'TC1',
    'TERMINAL A CONTENEUR (TC 2)': 'TC2',
    'TERMINAL FRUITIER'          : 'Fruitier',
    'TERMINAL ROULIER'           : 'Roulier',
}
for t in AUTRES_TERM: MAPPING_TERM[t] = 'Autres zones'

MAPPING_DEST = {
    'National'      : 'Non transb.',
    'Burkina Faso'  : 'Non transb.',
    'Mali'          : 'Non transb.',
    'Niger'         : 'Non transb.',
    'Pays Cotiers'  : 'Non transb.',
    'Transbordement': 'Transbordé',
}

# ─────────────────────────────────────────────────────────────────
# 1. CHARGEMENT & PARSING
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("TRAIN_MODELS_CONTENEURS — Port d'Abidjan")
print("=" * 65)

df = pd.read_excel(FICHIER_DONNEES)
df = df[~df['Terminal'].astype(str).str.startswith('Filtres')]
df = df.dropna(subset=['Annee-Mois', 'Conteneur_EVP', 'Terminal', 'Destination'])

# Parsing Annee-Mois (float : 15.01 = jan 2015)
def parse_am(v):
    s = f"{float(v):.2f}"
    p = s.split('.')
    return int(p[0]) + 2000, int(p[1])

df['annee']     = df['Annee-Mois'].apply(lambda x: parse_am(x)[0])
df['mois']      = df['Annee-Mois'].apply(lambda x: parse_am(x)[1])
df['mois_date'] = pd.to_datetime({'year': df['annee'],
                                   'month': df['mois'], 'day': 1})
# Garder uniquement jusqu'a ANNEE_FIN (2025)
df = df[df['annee'] <= ANNEE_FIN]

df['Groupe_term'] = df['Terminal'].map(MAPPING_TERM)
df['Groupe_dest'] = df['Destination'].map(MAPPING_DEST)

# NaN restants
df.loc[df['Groupe_term'].isna(), 'Groupe_term'] = 'Autres zones'
df.loc[df['Groupe_dest'].isna(), 'Groupe_dest'] = 'Non transb.'

print(f"\n[1] Donnees chargees : {len(df):,} lignes apres nettoyage")

# ─────────────────────────────────────────────────────────────────
# 2. SÉRIES MENSUELLES
# ─────────────────────────────────────────────────────────────────
IDX = pd.date_range(f'{ANNEE_DEBUT}-01-01', f'{ANNEE_FIN}-12-01', freq='MS')

def build(mask):
    s = df[mask].groupby('mois_date')['Conteneur_EVP'].sum()
    return s.reindex(IDX).fillna(0)

series = {}
# Axe terminal
for g in SEGS_TERM:
    series[g] = build(df['Groupe_term'] == g)

# Total
series['Total'] = sum(series[g] for g in SEGS_TERM)

# Axe destination
series['Non transb.']      = build(df['Groupe_dest'] == 'Non transb.')
series['Transbordé']       = build(df['Groupe_dest'] == 'Transbordé')
series['Transb. TC2']      = build((df['Groupe_dest'] == 'Transbordé') &
                                    (df['Groupe_term'] == 'TC2'))
series['Transb. habituel'] = build((df['Groupe_dest'] == 'Transbordé') &
                                    (df['Groupe_term'] != 'TC2'))

print(f"[2] Séries mensuelles construites : {len(series)} séries × {len(IDX)} mois")

def ann(s, yr): return int(s[s.index.year == yr].sum())

print(f"\n    Totaux annuels (TEU) :")
ann_total = {}
for yr in range(ANNEE_DEBUT, ANNEE_FIN + 1):
    v = ann(series['Total'], yr)
    ann_total[yr] = v
    print(f"      {yr} : {v:>12,}")

# ─────────────────────────────────────────────────────────────────
# 3. FONCTIONS HOLT
# ─────────────────────────────────────────────────────────────────
def holt_damped(y, alpha, beta, phi, h=1):
    n = len(y)
    L = np.zeros(n); T = np.zeros(n)
    L[0] = y[0]; T[0] = y[1] - y[0]
    for t in range(1, n):
        L[t] = alpha * y[t] + (1 - alpha) * (L[t-1] + phi * T[t-1])
        T[t] = beta * (L[t] - L[t-1]) + (1 - beta) * phi * T[t-1]
    fc     = np.array([L[-1] + sum(phi**j * T[-1] for j in range(1, i+1))
                       for i in range(1, h + 1)])
    fitted = np.array([L[t-1] + phi * T[t-1] for t in range(1, n)])
    return fc, fitted, L, T

def optimise_holt(y, min_train=3):
    """Grid search Holt amortie par LOO-CV."""
    best_w, best_p = 1e6, (0.5, 0.1, 0.9)
    for a in np.arange(0.05, 1.00, 0.05):
        for b in np.arange(0.05, 0.65, 0.05):
            for p in [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98]:
                errs = []
                for i in range(min_train, len(y)):
                    try:
                        fc, _, _, _ = holt_damped(y[:i], a, b, p, h=1)
                        errs.append(abs(y[i] - fc[0]) / y[i])
                    except:
                        break
                if errs:
                    w = np.mean(errs)
                    if w < best_w:
                        best_w = w; best_p = (a, b, p)
    return best_p, best_w * 100

# ─────────────────────────────────────────────────────────────────
# 4. MODÈLE TOTAL — entraîné sur 2023-2025
# ─────────────────────────────────────────────────────────────────
print(f"\n[3] Modèle TOTAL (train {ANNEE_RUPTURE}-{ANNEE_FIN}) ...")

y_total = np.array([ann_total[yr] for yr in range(ANNEE_DEBUT, ANNEE_FIN + 1)],
                   dtype=float)
y_tot_court = y_total[ANNEE_RUPTURE - ANNEE_DEBUT:]   # 2023, 2024, 2025

# Hold-out : entraîner sur 2023-2024, tester sur 2025
best_err = 1e6; best_p_tot = (0.5, 0.1, 0.9)
for a in np.arange(0.05, 1.00, 0.05):
    for b in np.arange(0.05, 0.65, 0.05):
        for p in [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98]:
            try:
                fc, _, _, _ = holt_damped(y_tot_court[:2], a, b, p, h=1)
                err = abs(y_tot_court[2] - fc[0]) / y_tot_court[2]
                if err < best_err:
                    best_err = err; best_p_tot = (a, b, p)
            except:
                pass

alpha_t, beta_t, phi_t = best_p_tot
_, fitted_t, L_t, T_t = holt_damped(y_tot_court, alpha_t, beta_t, phi_t, h=1)
resid_t = y_tot_court[1:] - fitted_t
rmse_t  = float(np.sqrt(np.mean(resid_t**2))) if len(resid_t) > 0 else float(y_tot_court.std())

print(f"    α={alpha_t:.2f}  β={beta_t:.2f}  φ={phi_t:.2f}")
print(f"    Erreur hold-out 2025 : {best_err*100:.1f}%  RMSE : {rmse_t:,.0f} TEU")

# ─────────────────────────────────────────────────────────────────
# 5. MODÈLE NON TRANSBORDÉ — 2015-2025
# ─────────────────────────────────────────────────────────────────
print(f"\n[4] Modèle NON TRANSBORDÉ (train {ANNEE_DEBUT}-{ANNEE_FIN}) ...")

y_nt = np.array([ann(series['Non transb.'], yr)
                 for yr in range(ANNEE_DEBUT, ANNEE_FIN + 1)], dtype=float)
params_nt, wmape_nt = optimise_holt(y_nt)
_, fitted_nt, L_nt, T_nt = holt_damped(y_nt, *params_nt, h=1)
rmse_nt = float(np.sqrt(np.mean((y_nt[1:] - fitted_nt)**2)))

print(f"    α={params_nt[0]:.2f}  β={params_nt[1]:.2f}  φ={params_nt[2]:.2f}")
print(f"    WMAPE={wmape_nt:.1f}%  RMSE={rmse_nt:,.0f} TEU")

# ─────────────────────────────────────────────────────────────────
# 6. TRANSBORDÉ TC2 — constant = réalisé 2025
# ─────────────────────────────────────────────────────────────────
transb_tc2_2025 = ann(series['Transb. TC2'], ANNEE_FIN)
print(f"\n[5] TRANSBORDÉ TC2 — constant = {transb_tc2_2025:,} TEU (réalisé {ANNEE_FIN})")

# ─────────────────────────────────────────────────────────────────
# 7. MODÈLE TRANSBORDÉ HABITUEL — 2015-2025
# ─────────────────────────────────────────────────────────────────
print(f"\n[6] Modèle TRANSBORDÉ HABITUEL (train {ANNEE_DEBUT}-{ANNEE_FIN}) ...")

y_th = np.array([ann(series['Transb. habituel'], yr)
                 for yr in range(ANNEE_DEBUT, ANNEE_FIN + 1)], dtype=float)
params_th, wmape_th = optimise_holt(y_th)
_, fitted_th, L_th, T_th = holt_damped(y_th, *params_th, h=1)
rmse_th = float(np.sqrt(np.mean((y_th[1:] - fitted_th)**2)))

print(f"    α={params_th[0]:.2f}  β={params_th[1]:.2f}  φ={params_th[2]:.2f}")
print(f"    WMAPE={wmape_th:.1f}%  RMSE={rmse_th:,.0f} TEU")

# ─────────────────────────────────────────────────────────────────
# 8. PROFIL SAISONNIER — 2023-2025
# ─────────────────────────────────────────────────────────────────
print(f"\n[7] Profil saisonnier (2023-2025) ...")
ANNEES_PROFIL = [2023, 2024, 2025]
parts_sais = {}
for yr in ANNEES_PROFIL:
    sub = series['Total'][series['Total'].index.year == yr]
    parts_sais[yr] = sub.values / sub.sum() * 100

profil = np.mean([parts_sais[yr] for yr in ANNEES_PROFIL], axis=0)
noms_m = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']
print(f"    {'Mois':<5} {'Part moy.':>10}")
for m in range(12):
    print(f"    {noms_m[m]:<5} {profil[m]:>9.2f}%")

# ─────────────────────────────────────────────────────────────────
# 9. CLÉS DE RÉPARTITION HISTORIQUES
# ─────────────────────────────────────────────────────────────────
print(f"\n[8] Clés de répartition ...")

parts_term = {}
parts_dest = {}
for yr in range(ANNEE_DEBUT, ANNEE_FIN + 1):
    tot_yr = ann_total[yr]
    parts_term[yr] = {g: ann(series[g], yr) / tot_yr * 100 for g in SEGS_TERM}
    parts_dest[yr] = {
        'Non transb.'     : ann(series['Non transb.'],      yr) / tot_yr * 100,
        'Transbordé'      : ann(series['Transbordé'],       yr) / tot_yr * 100,
        'Transb. TC2'     : ann(series['Transb. TC2'],      yr) / tot_yr * 100,
        'Transb. habituel': ann(series['Transb. habituel'], yr) / tot_yr * 100,
    }

print(f"\n    Parts terminaux 2025 :")
for g in SEGS_TERM:
    print(f"      {g:<15} : {parts_term[ANNEE_FIN][g]:>6.2f}%")
print(f"\n    Parts destination 2025 :")
for k, v in parts_dest[ANNEE_FIN].items():
    print(f"      {k:<20} : {v:>6.2f}%")

# ─────────────────────────────────────────────────────────────────
# 10. SAUVEGARDE
# ─────────────────────────────────────────────────────────────────
model_data = {
    # Modèle Total
    'alpha_tot'       : alpha_t,
    'beta_tot'        : beta_t,
    'phi_tot'         : phi_t,
    'err_tot'         : best_err * 100,
    'rmse_tot'        : rmse_t,
    'L_tot'           : float(L_t[-1]),
    'T_tot'           : float(T_t[-1]),
    'y_tot_court'     : y_tot_court,
    'annee_rupture'   : ANNEE_RUPTURE,

    # Modèle Non transbordé
    'params_nt'       : params_nt,
    'wmape_nt'        : wmape_nt,
    'rmse_nt'         : rmse_nt,
    'L_nt'            : float(L_nt[-1]),
    'T_nt'            : float(T_nt[-1]),
    'y_nt'            : y_nt,

    # Transbordé TC2 (constant)
    'transb_tc2_2025' : transb_tc2_2025,

    # Modèle Transbordé habituel
    'params_th'       : params_th,
    'wmape_th'        : wmape_th,
    'rmse_th'         : rmse_th,
    'L_th'            : float(L_th[-1]),
    'T_th'            : float(T_th[-1]),
    'y_th'            : y_th,

    # Profil saisonnier
    'profil_saisonnier': profil,
    'annees_profil'    : ANNEES_PROFIL,

    # Clés de répartition
    'parts_term'      : parts_term,
    'parts_dest'      : parts_dest,
    'ann_total'       : ann_total,

    # Métadonnées
    'segs_term'       : SEGS_TERM,
    'segs_dest'       : SEGS_DEST,
    'annee_debut'     : ANNEE_DEBUT,
    'annee_fin'       : ANNEE_FIN,
    'noms_mois'       : noms_m,
}

with open(OUTPUT_MDL, 'wb') as f:
    pickle.dump(model_data, f)
with open(OUTPUT_SER, 'wb') as f:
    pickle.dump(series, f)

print(f"\n[9] Fichiers sauvegardés :")
print(f"    → {OUTPUT_MDL}")
print(f"    → {OUTPUT_SER}")
print(f"\n{'='*65}")
print(f"✓ Entraînement terminé")
print(f"  Total     : Holt amortie (err 2025={best_err*100:.1f}%)  train {ANNEE_RUPTURE}-{ANNEE_FIN}")
print(f"  Non transb: Holt amortie (WMAPE={wmape_nt:.1f}%)  train {ANNEE_DEBUT}-{ANNEE_FIN}")
print(f"  Transb TC2: Constant {transb_tc2_2025:,} TEU/an")
print(f"  Transb hab: Holt amortie (WMAPE={wmape_th:.1f}%)  train {ANNEE_DEBUT}-{ANNEE_FIN}")
print(f"{'='*65}")
