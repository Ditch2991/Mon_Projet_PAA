"""
train_models_escales.py — Entraînement modèle escales Port d'Abidjan
=====================================================================
Modèle : Holt amortie (double lissage exponentiel avec tendance amortie)
Approche : Top-down annuel → ventilation mensuelle par profil saisonnier
           → répartition par terminal (clés N-1)

Usage  : python train_models_escales.py
Sortie : models_escales.pkl  (modèle + paramètres + profil + clés)
         series_escales.pkl  (séries mensuelles historiques)
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────────
FICHIER_DONNEES = "data_Escales.xlsx"
OUTPUT_MDL      = "models_escales.pkl"
OUTPUT_SER      = "series_escales.pkl"

ANNEE_DEBUT     = 2015
ANNEE_FIN       = 2025
ANNEES_PROFIL   = [2023, 2024, 2025]   # années sans données manquantes

SEGS = ['TC1', 'TC2', 'Céréalier', 'Fruitier', 'Minéralier',
        'Pétrolier', 'Roulier', 'Quai Nord', 'Quai Ouest', 'Autres zones']

AUTRES_ZONES = ['AZITO','CARENA','COFFRE','MOUILLAGE','Mouillage Sogip',
                'PHILIPPS','Quai SOGIP','SIVENG','SOCOPAO','SUD','Siap',
                'TERMINAL PECHE']

MAPPING_TERMINAL = {
    'TERMINAL A CONTENEUR (TC 1)': 'TC1',
    'TERMINAL A CONTENEUR (TC 2)': 'TC2',
    'TERMINAL CEREALIER'         : 'Céréalier',
    'TERMINAL FRUITIER'          : 'Fruitier',
    'TERMINAL MINERALIER'        : 'Minéralier',
    'TERMINAL PETROLIER'         : 'Pétrolier',
    'TERMINAL ROULIER'           : 'Roulier',
    'Quai NORD (Q1-Q5)'          : 'Quai Nord',
    'Quai OUEST (Q6-Q10)'        : 'Quai Ouest',
}
for t in AUTRES_ZONES:
    MAPPING_TERMINAL[t] = 'Autres zones'

# ─────────────────────────────────────────────────────────────────
# 1. CHARGEMENT & PARSING
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("TRAIN_MODELS_ESCALES — Port d'Abidjan")
print("=" * 60)

df = pd.read_excel(FICHIER_DONNEES)
# Supprimer les lignes de filtre (différents formats possibles)
df = df[~df['Terminal'].astype(str).str.startswith('Filtres')]
df = df[df['Terminal'] != 'Aucun filtre appliqué']
df = df.dropna(subset=['Annee-Mois', 'Nb Escales'])

# Parsing robuste : Annee-Mois est float (ex: 15.1 = octobre 2015)
def parse_am(v):
    s = f"{float(v):.2f}"   # 15.1 → "15.10"
    p = s.split('.')
    return int(p[0]) + 2000, int(p[1])

df['annee'] = df['Annee-Mois'].apply(lambda x: parse_am(x)[0])
df['mois']  = df['Annee-Mois'].apply(lambda x: parse_am(x)[1])
df['date']  = pd.to_datetime({'year': df['annee'], 'month': df['mois'], 'day': 1})
# Garder seulement jusqu'à ANNEE_FIN
df = df[df['annee'] <= ANNEE_FIN]

df['Groupe'] = df['Terminal'].map(MAPPING_TERMINAL)
# Terminal NaN = escales non attribuées → incluses dans Autres zones
df.loc[df['Groupe'].isna(), 'Groupe'] = 'Autres zones'

print(f"\n[1] Données chargées : {len(df)} lignes après nettoyage")

# ─────────────────────────────────────────────────────────────────
# 2. CONSTRUCTION DES SÉRIES MENSUELLES
# ─────────────────────────────────────────────────────────────────
IDX = pd.date_range(f'{ANNEE_DEBUT}-01-01', f'{ANNEE_FIN}-12-01', freq='MS')

def build_serie(grp):
    s = df[df['Groupe'] == grp].groupby('date')['Nb Escales'].sum()
    return s.reindex(IDX).fillna(0)

series_mens = {g: build_serie(g) for g in SEGS}
total_mens  = sum(series_mens[g] for g in SEGS)
series_mens['Total'] = total_mens

print(f"[2] Séries mensuelles construites : {len(SEGS)+1} séries × {len(IDX)} mois")

# Vérification totaux annuels
print(f"\n    Total annuel escales :")
ann_total = {}
for yr in range(ANNEE_DEBUT, ANNEE_FIN + 1):
    v = int(total_mens[total_mens.index.year == yr].sum())
    ann_total[yr] = v
    print(f"      {yr} : {v:>5}")

# ─────────────────────────────────────────────────────────────────
# 3. MODÈLE HOLT AMORTIE SUR TOTAL ANNUEL
# ─────────────────────────────────────────────────────────────────
print(f"\n[3] Optimisation Holt amortie (grid search α, β, φ) ...")

y = np.array([ann_total[yr] for yr in range(ANNEE_DEBUT, ANNEE_FIN + 1)], dtype=float)

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

def wmape_loo(params, y):
    a, b, p = params
    if not (0 < a < 1 and 0 < b < 1 and 0.8 <= p < 1):
        return 1e6
    errs = []
    for i in range(3, len(y)):
        try:
            fc, _, _, _ = holt_damped_forecast(y[:i], a, b, p, h=1)
            errs.append(abs(y[i] - fc[0]) / y[i])
        except:
            return 1e6
    return np.mean(errs)

best_w, best_params = 1e6, (0.3, 0.1, 0.9)
for a in np.arange(0.10, 1.00, 0.05):
    for b in np.arange(0.05, 0.65, 0.05):
        for p in [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98]:
            w = wmape_loo((a, b, p), y)
            if w < best_w:
                best_w = w; best_params = (a, b, p)

alpha, beta, phi = best_params
print(f"    Paramètres optimaux : α={alpha:.2f}  β={beta:.2f}  φ={phi:.2f}")
print(f"    WMAPE (LOO-CV)       : {best_w * 100:.1f}%")

# Fitted values & résidus
_, fitted, L_opt, T_opt = holt_damped_forecast(y, alpha, beta, phi, h=1)
resid = y[1:] - fitted
rmse  = np.sqrt(np.mean(resid**2))
print(f"    RMSE                 : {rmse:.1f} escales")

# ─────────────────────────────────────────────────────────────────
# 4. PROFIL SAISONNIER (2023-2025)
# ─────────────────────────────────────────────────────────────────
print(f"\n[4] Calcul du profil saisonnier sur {ANNEES_PROFIL} ...")

parts_par_annee = {}
for yr in ANNEES_PROFIL:
    sub = total_mens[total_mens.index.year == yr]
    parts_par_annee[yr] = (sub.values / sub.sum() * 100)

profil_saisonnier = np.mean([parts_par_annee[yr] for yr in ANNEES_PROFIL], axis=0)

noms_m = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']
print(f"    {'Mois':<5} {'Part moy.':>10}")
for m in range(12):
    print(f"    {noms_m[m]:<5} {profil_saisonnier[m]:>9.2f}%")
print(f"    Somme : {profil_saisonnier.sum():.2f}%")

# ─────────────────────────────────────────────────────────────────
# 5. CLÉS DE RÉPARTITION PAR TERMINAL (historique + 2025)
# ─────────────────────────────────────────────────────────────────
print(f"\n[5] Calcul des clés de répartition par terminal ...")

parts_terminaux = {}
for yr in range(ANNEE_DEBUT, ANNEE_FIN + 1):
    tot_yr = total_mens[total_mens.index.year == yr].sum()
    parts_terminaux[yr] = {
        g: series_mens[g][series_mens[g].index.year == yr].sum() / tot_yr * 100
        for g in SEGS
    }

print(f"    Parts 2025 (clé initiale pour 2026) :")
for g in SEGS:
    print(f"      {g:<15} : {parts_terminaux[2025][g]:>5.2f}%")

# ─────────────────────────────────────────────────────────────────
# 6. SAUVEGARDE
# ─────────────────────────────────────────────────────────────────
model_data = {
    # Modèle Holt amortie
    'alpha'            : alpha,
    'beta'             : beta,
    'phi'              : phi,
    'wmape'            : best_w * 100,
    'rmse'             : rmse,
    'L_final'          : L_opt[-1],
    'T_final'          : T_opt[-1],

    # Données historiques annuelles
    'ann_total'        : ann_total,
    'y_train'          : y,
    'fitted'           : fitted,
    'resid'            : resid,

    # Profil saisonnier
    'profil_saisonnier': profil_saisonnier,
    'annees_profil'    : ANNEES_PROFIL,

    # Clés de répartition
    'parts_terminaux'  : parts_terminaux,
    'segs'             : SEGS,
    'annee_debut'      : ANNEE_DEBUT,
    'annee_fin'        : ANNEE_FIN,

}

with open(OUTPUT_MDL, 'wb') as f:
    pickle.dump(model_data, f)

with open(OUTPUT_SER, 'wb') as f:
    pickle.dump(series_mens, f)

print(f"\n[6] Fichiers sauvegardés :")
print(f"    → {OUTPUT_MDL}")
print(f"    → {OUTPUT_SER}")

print(f"\n{'='*60}")
print(f"✓ Entraînement terminé — WMAPE = {best_w*100:.1f}%")
print(f"{'='*60}")
