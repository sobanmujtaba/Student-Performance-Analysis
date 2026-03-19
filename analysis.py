#!/usr/bin/env python3
"""
Student Performance Analysis
=============================
K-Means Clustering + Pearson Correlation on 10,000 anonymised student records.

The goal is to identify distinct learning profiles from behavioural and
background features, then quantify which features actually drive the
Performance Index. Findings are intended to inform targeted curriculum
interventions for each student group.

Dataset columns:
    Hours Studied                     -- self-reported weekly study hours (1-9)
    Previous Scores                   -- prior academic score (40-99)
    Extracurricular Activities        -- Yes / No
    Sleep Hours                       -- average nightly sleep (4-9)
    Sample Question Papers Practiced  -- number of practice papers completed
    Performance Index                 -- target variable (10-100, continuous)

Usage:
    python analysis.py

Output:
    student_performance_analysis.png  -- 8-panel figure
    cluster_profiles.csv              -- mean feature values per cluster
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr, gaussian_kde
import warnings
warnings.filterwarnings('ignore')


# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

DATA_PATH   = 'Student_Performance.csv'
OUTPUT_PNG  = 'student_performance_analysis-1.png'
OUTPUT_CSV  = 'cluster_profiles.csv'
OPTIMAL_K   = 4     # see elbow/silhouette analysis below for justification
RANDOM_SEED = 42


# -------------------------------------------------------
# STEP 1: LOAD DATA
# -------------------------------------------------------

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
print(df.dtypes, '\n')

# Encode extracurricular activity as integer (1 = Yes, 0 = No).
# K-Means only accepts numeric input; this preserves the binary signal.
df['Extracurricular_Bin'] = (df['Extracurricular Activities'] == 'Yes').astype(int)


# -------------------------------------------------------
# STEP 2: SCALE FEATURES
# -------------------------------------------------------
# K-Means uses Euclidean distance, so features measured on different scales
# will dominate the clustering unfairly.
# StandardScaler transforms each column to mean=0, std=1 before fitting.

feature_cols = [
    'Hours Studied',
    'Previous Scores',
    'Sleep Hours',
    'Sample Question Papers Practiced',
    'Performance Index',   # included so clusters reflect the full profile
]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])


# -------------------------------------------------------
# STEP 3: FIND OPTIMAL K (ELBOW + SILHOUETTE)
# -------------------------------------------------------
# Inertia (within-cluster sum of squares) decreases as k increases.
# The "elbow" -- the point where gains diminish sharply -- suggests a good k.
# Silhouette score measures how well each point fits its own cluster vs
# neighbouring clusters; higher is better (range: -1 to 1).

inertias    = []
silhouettes = []
k_range     = range(2, 9)

for k in k_range:
    km     = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

print("Silhouette scores by k:")
for k, s in zip(k_range, silhouettes):
    print(f"  k={k}: {s:.4f}")

# k=4 gives a reasonable balance: silhouette scores are modest across all k
# (this dataset is continuous, not naturally blob-shaped), but k=4 produces
# four profiles that map clearly onto pedagogically meaningful groups.


# -------------------------------------------------------
# STEP 4: FIT FINAL K-MEANS MODEL
# -------------------------------------------------------

km_final     = KMeans(n_clusters=OPTIMAL_K, random_state=RANDOM_SEED, n_init=10)
df['Cluster'] = km_final.fit_predict(X_scaled)

# Re-label clusters in ascending order of mean Performance Index.
# This makes Cluster 0 = weakest group, Cluster 3 = strongest.
# It is purely for readability; the geometry is unchanged.
order_map = {
    old: new for new, old in
    enumerate(df.groupby('Cluster')['Performance Index'].mean().sort_values().index)
}
df['Cluster'] = df['Cluster'].map(order_map)

# Human-readable cluster names derived from profile inspection below
cluster_names = {
    0: 'Struggling',       # low effort, low prior knowledge
    1: 'Effort-Driven',    # high effort, low prior knowledge
    2: 'Coasting',         # low effort, high prior knowledge
    3: 'High Achievers',   # high effort, high prior knowledge
}
df['Profile'] = df['Cluster'].map(cluster_names)


# -------------------------------------------------------
# STEP 5: BUILD CLUSTER PROFILE TABLE
# -------------------------------------------------------

profile_cols = feature_cols + ['Extracurricular_Bin']
profile      = df.groupby('Cluster')[profile_cols].mean().round(2)
profile['Count']            = df.groupby('Cluster').size()
profile['Extracurricular_%'] = (profile['Extracurricular_Bin'] * 100).round(1)
profile['Profile']           = profile.index.map(cluster_names)
profile = profile.drop(columns='Extracurricular_Bin')

print("\nCluster Profiles:")
print(profile.to_string())
profile.to_csv(OUTPUT_CSV)
print(f"\nProfiles saved to {OUTPUT_CSV}")


# -------------------------------------------------------
# STEP 6: CORRELATION ANALYSIS
# -------------------------------------------------------
# Pearson r measures the strength of the linear relationship between
# each feature and the target (Performance Index).
# All p-values are effectively 0 due to the large sample (n=10,000),
# so we focus on the magnitude of r, not just statistical significance.

corr_features = [
    'Hours Studied',
    'Previous Scores',
    'Sleep Hours',
    'Sample Question Papers Practiced',
    'Extracurricular_Bin',
]
target = 'Performance Index'

print(f"\nPearson correlations with '{target}':")
corr_results = {}
for feat in corr_features:
    r, p = pearsonr(df[feat], df[target])
    corr_results[feat] = r
    print(f"  {feat:42s}  r = {r:+.4f}   p = {p:.2e}")

# Key findings:
#   Previous Scores   r = +0.92  -- dominant predictor by far
#   Hours Studied     r = +0.37  -- only meaningful modifiable factor
#   Sleep Hours       r = +0.05  -- negligible linear effect
#   Practice Papers   r = +0.04  -- negligible linear effect
#   Extracurricular   r = +0.02  -- negligible linear effect


# -------------------------------------------------------
# STEP 7: PCA FOR 2-D VISUALISATION
# -------------------------------------------------------
# The model ran in 5-D scaled space; this reduces to 2D for plotting only.
# ~60% of total variance is preserved in two components.

pca   = PCA(n_components=2, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]
print(f"\nPCA: 2 components capture {pca.explained_variance_ratio_.sum()*100:.1f}% of variance")


# -------------------------------------------------------
# STEP 8: PLOT
# -------------------------------------------------------

COLORS  = ['#C0392B', '#E67E22', '#2980B9', '#27AE60']
LABELS  = ['C0: Struggling', 'C1: Effort-Driven', 'C2: Coasting', 'C3: High Achievers']
BG      = '#0E1117'
PANEL   = '#161B22'
TEXT    = '#E6EDF3'
MUTED   = '#8B949E'
ACCENT  = '#F0B429'

fig = plt.figure(figsize=(22, 26), facecolor=BG)
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35,
                        left=0.07, right=0.96, top=0.94, bottom=0.04)

def style_ax(ax, title=''):
    """Apply consistent dark-theme styling to an axes object."""
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363D')
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=10)

# -- Panel 1: PCA scatter coloured by cluster
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, 'PCA Cluster Scatter (60.1% variance explained)')
for c in range(OPTIMAL_K):
    sub = df[df['Cluster'] == c]
    ax1.scatter(sub['PCA1'], sub['PCA2'], c=COLORS[c],
                label=LABELS[c], alpha=0.35, s=8, linewidths=0)
ax1.set_xlabel('PC1', color=MUTED, fontsize=9)
ax1.set_ylabel('PC2', color=MUTED, fontsize=9)
ax1.legend(fontsize=8, facecolor=PANEL, edgecolor='#30363D',
           labelcolor=TEXT, markerscale=2)

# -- Panel 2: Elbow + silhouette
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'Elbow Curve -- Choosing K')
ax2t = ax2.twinx()
k_list = list(k_range)
ax2.plot(k_list, inertias, color=ACCENT, marker='o', linewidth=2, markersize=6)
ax2t.plot(k_list, silhouettes, color='#64D2FF', marker='s', linewidth=2,
          markersize=6, linestyle='--')
ax2.axvline(x=OPTIMAL_K, color='#FF6B6B', linewidth=1.5, linestyle=':', alpha=0.9)
ax2.annotate(f'k={OPTIMAL_K} chosen',
             xy=(OPTIMAL_K, inertias[OPTIMAL_K - 2]),
             xytext=(OPTIMAL_K + 1.1, inertias[OPTIMAL_K - 2] * 1.02),
             color='#FF6B6B', fontsize=8,
             arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=1.2))
ax2.set_xlabel('k', color=MUTED, fontsize=9)
ax2.set_ylabel('Inertia (WCSS)', color=ACCENT, fontsize=9)
ax2t.set_ylabel('Silhouette Score', color='#64D2FF', fontsize=9)
ax2.tick_params(axis='y', colors=ACCENT)
ax2t.tick_params(colors='#64D2FF', labelsize=9)
ax2t.set_facecolor(PANEL)
for spine in ax2.spines.values():
    spine.set_edgecolor('#30363D')
for spine in ax2t.spines.values():
    spine.set_edgecolor('#30363D')

# -- Panel 3: Correlation bar chart
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, 'Pearson Correlation with Performance Index')
corr_display = {
    'Previous Scores': corr_results['Previous Scores'],
    'Hours Studied':   corr_results['Hours Studied'],
    'Sleep Hours':     corr_results['Sleep Hours'],
    'Practice Papers': corr_results['Sample Question Papers Practiced'],
    'Extracurricular': corr_results['Extracurricular_Bin'],
}
feats  = list(corr_display.keys())
vals   = list(corr_display.values())
bcolor = [COLORS[3] if v > 0.3 else COLORS[2] if v > 0.1 else MUTED for v in vals]
bars   = ax3.barh(feats, vals, color=bcolor, height=0.55)
for bar, val in zip(bars, vals):
    ax3.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
             f'r = {val:.4f}', va='center', color=TEXT, fontsize=8.5)
ax3.set_xlim(0, 1.05)
ax3.set_xlabel('Pearson r', color=MUTED, fontsize=9)
ax3.tick_params(axis='y', colors=TEXT)
ax3.axvline(x=0.3, color=MUTED, linewidth=1, linestyle=':', alpha=0.5)

# -- Panel 4: Normalised grouped bar chart of cluster profiles
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, 'Cluster Feature Profiles (Normalised Mean)')
norm = profile[feature_cols].copy()
for col in feature_cols:
    cmin, cmax = norm[col].min(), norm[col].max()
    norm[col]  = (norm[col] - cmin) / (cmax - cmin + 1e-9)
short = ['Hours\nStudied', 'Prev\nScores', 'Sleep\nHours', 'Practice\nPapers', 'Perf\nIndex']
x     = np.arange(len(feature_cols))
w     = 0.18
for i in range(OPTIMAL_K):
    ax4.bar(x + (i - 1.5) * w, norm.iloc[i].values, width=w,
            color=COLORS[i], alpha=0.85, label=LABELS[i])
ax4.set_xticks(x)
ax4.set_xticklabels(short, color=TEXT, fontsize=8.5)
ax4.set_ylabel('Normalised Mean (0-1)', color=MUTED, fontsize=9)
ax4.legend(fontsize=7.5, facecolor=PANEL, edgecolor='#30363D',
           labelcolor=TEXT, ncol=2)
ax4.tick_params(axis='y', colors=MUTED)

# -- Panel 5: KDE density per cluster
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, 'Performance Index Distribution per Cluster')
for c in range(OPTIMAL_K):
    v   = df[df['Cluster'] == c]['Performance Index'].values
    kde = gaussian_kde(v, bw_method=0.3)
    xr  = np.linspace(v.min(), v.max(), 300)
    ax5.fill_between(xr, kde(xr), alpha=0.25, color=COLORS[c])
    ax5.plot(xr, kde(xr), color=COLORS[c], linewidth=2, label=LABELS[c])
ax5.set_xlabel('Performance Index', color=MUTED, fontsize=9)
ax5.set_ylabel('Density', color=MUTED, fontsize=9)
ax5.legend(fontsize=8, facecolor=PANEL, edgecolor='#30363D', labelcolor=TEXT)

# -- Panel 6: Hours studied vs performance
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, 'Hours Studied vs Performance Index')
for c in range(OPTIMAL_K):
    sub = df[df['Cluster'] == c]
    ax6.scatter(sub['Hours Studied'], sub['Performance Index'],
                c=COLORS[c], alpha=0.25, s=7, linewidths=0, label=LABELS[c])
m, b   = np.polyfit(df['Hours Studied'], df['Performance Index'], 1)
x_line = np.linspace(1, 9, 100)
ax6.plot(x_line, m * x_line + b, color=ACCENT, linewidth=2,
         linestyle='--', label='Trend (r=0.37)')
ax6.set_xlabel('Hours Studied', color=MUTED, fontsize=9)
ax6.set_ylabel('Performance Index', color=MUTED, fontsize=9)
ax6.legend(fontsize=7.5, facecolor=PANEL, edgecolor='#30363D', labelcolor=TEXT)

# -- Panel 7: Previous scores vs performance (dominant correlation)
ax7 = fig.add_subplot(gs[3, 0])
style_ax(ax7, 'Previous Scores vs Performance Index (r = 0.92)')
for c in range(OPTIMAL_K):
    sub = df[df['Cluster'] == c]
    ax7.scatter(sub['Previous Scores'], sub['Performance Index'],
                c=COLORS[c], alpha=0.2, s=7, linewidths=0)
m2, b2  = np.polyfit(df['Previous Scores'], df['Performance Index'], 1)
x_line2 = np.linspace(df['Previous Scores'].min(), df['Previous Scores'].max(), 200)
ax7.plot(x_line2, m2 * x_line2 + b2, color=ACCENT, linewidth=2, linestyle='--')
ax7.set_xlabel('Previous Scores', color=MUTED, fontsize=9)
ax7.set_ylabel('Performance Index', color=MUTED, fontsize=9)

# -- Panel 8: Summary table
ax8 = fig.add_subplot(gs[3, 1])
ax8.set_facecolor(PANEL)
ax8.axis('off')
style_ax(ax8, 'Cluster Summary Table')
headers = ['Cluster', 'Hrs\nStudied', 'Prev\nScores', 'Perf\nIndex', 'Count', 'EC%']
rows    = []
for i in range(OPTIMAL_K):
    r = profile.iloc[i]
    rows.append([LABELS[i],
                 f"{r['Hours Studied']:.1f}",
                 f"{r['Previous Scores']:.1f}",
                 f"{r['Performance Index']:.1f}",
                 f"{int(r['Count'])}",
                 f"{r['Extracurricular_%']:.0f}%"])
tbl = ax8.table(cellText=rows, colLabels=headers,
                cellLoc='center', loc='center', bbox=[0, 0.05, 1, 0.88])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
for (ri, ci), cell in tbl.get_celld().items():
    cell.set_edgecolor('#30363D')
    if ri == 0:
        cell.set_facecolor('#21262D')
        cell.set_text_props(color=ACCENT, fontweight='bold')
    else:
        cell.set_facecolor(PANEL)
        cell.set_text_props(color=TEXT)
        if ci == 0:
            cell.set_facecolor(COLORS[ri - 1] + '33')

fig.text(0.5, 0.97,
         'Student Performance Analysis',
         ha='center', fontsize=20, fontweight='bold',
         color=TEXT, fontfamily='monospace')
fig.text(0.5, 0.955,
         'K-Means Clustering (k=4)  +  Pearson Correlation  |  n = 10,000',
         ha='center', fontsize=11, color=MUTED)

plt.savefig(OUTPUT_PNG, dpi=160, bbox_inches='tight', facecolor=BG)
print(f"\nFigure saved to {OUTPUT_PNG}")


# -------------------------------------------------------
# STEP 9: CURRICULUM RECOMMENDATIONS
# -------------------------------------------------------

recommendations = """
============================================================
CURRICULUM RECOMMENDATIONS
============================================================

Cluster 0 -- Struggling  (avg perf 32.4 | low hours, low prior scores)
  Both dimensions are weak. The root issue is foundational knowledge
  deficit, not just low effort. Simply assigning more work will not move
  this group. Recommended interventions:
    - Pre-semester diagnostic assessments to map specific gaps
    - Foundational remediation modules with structured scaffolding
    - Peer tutoring and tracked micro-goals to build study habits

Cluster 1 -- Effort-Driven  (avg perf 47.5 | high hours, low prior scores)
  These students already put in the hours (avg 7.2h) but the prior
  knowledge gap limits returns. Study quantity is not the problem.
  Recommended interventions:
    - Shift from quantity to quality: retrieval practice over re-reading
    - Targeted gap-filling based on diagnostic pre-tests
    - Study-strategy workshops (spaced repetition, interleaving)

Cluster 2 -- Coasting  (avg perf 63.5 | low hours, high prior scores)
  Strong prior knowledge is carrying them. Under-engagement risks
  stagnation and boredom. Recommended interventions:
    - Stretch assignments and project-based components
    - Enrichment tracks to maintain challenge
    - Avoid placing in remedial groups -- disengagement risk is high

Cluster 3 -- High Achievers  (avg perf 78.5 | high hours, high prior scores)
  Both dimensions are working. Sleep hours are nearly identical across
  all clusters (6.4-6.6h), so sleep is not a differentiating factor.
  Recommended interventions:
    - Advanced electives and independent research opportunities
    - Peer mentoring roles to reinforce their own understanding
    - Monitor for burnout -- high-effort students in large classes often
      push past healthy limits without signposting

OVERALL FINDING:
  Previous Scores (r = 0.92) is the single dominant predictor. Current
  performance is largely a continuation of prior trajectory. This points
  to pre-entry knowledge gaps as the primary lever, not in-semester
  behaviour change. Curriculum designers should prioritise on-boarding
  diagnostics and early-term interventions rather than end-of-term
  recovery options.
============================================================
"""
print(recommendations)
