import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. PENGUNCIAN DIREKTORI KERJA (ANTI-NYASAR)
# ==========================================
# Mendapatkan path absolut folder tempat skrip ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

print(f"Direktori kerja aktif terkunci di:\n-> {BASE_DIR}\n")

# Path absolut file database
db_path = os.path.join(BASE_DIR, "rfm_banking_database.db")

# Fallback jika file berada di path default
if not os.path.exists(db_path):
    fallback_path = r"D:\PORTOFOLIO\Data Analytic Projects\Customer Segmentation RFM Project Portofolio\rfm_banking_database.db"
    if os.path.exists(fallback_path):
        db_path = fallback_path

# ==========================================
# 1. KONEKSI & EKSTRAKSI DATA SQLITE
# ==========================================
print(f"Menghubungkan ke database: {os.path.basename(db_path)}...")
conn = sqlite3.connect(db_path)

query = """
SELECT 
    CustomerID,
    recency_days,
    frequency,
    monetary
FROM customer_rfm_summary;
"""

df_rfm = pd.read_sql_query(query, conn)
conn.close()

print(f"Data berhasil diekstraksi: {df_rfm.shape[0]:,} nasabah terdaftar.\n")

# ==========================================
# 2. PREPROCESSING & TRANSFORMASI DATA
# ==========================================
# Mengurangi skewness ekstrem pada distribusi perbankan dengan log1p
features = ['recency_days', 'frequency', 'monetary']
df_transformed = np.log1p(df_rfm[features])

# Standardisasi fitur (Z-score: Mean = 0, Std = 1)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(df_transformed)

# ==========================================
# 3. EVALUASI K OPTIMAL (ELBOW & SILHOUETTE)
# ==========================================
# Menggunakan sampel representatif 30.000 data untuk efisiensi komputasi
sample_indices = np.random.RandomState(42).choice(len(rfm_scaled), size=30000, replace=False)
rfm_sample = rfm_scaled[sample_indices]

k_range = range(2, 7)
inertia_list = []
silhouette_list = []

print("Mengevaluasi K optimal (K=2 hingga K=6)...")
for k in k_range:
    kmeans_eval = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans_eval.fit(rfm_sample)
    inertia_list.append(kmeans_eval.inertia_)
    
    sil_score = silhouette_score(rfm_sample, kmeans_eval.labels_, sample_size=5000, random_state=42)
    silhouette_list.append(sil_score)
    print(f"-> K={k} | WCSS/Inertia: {kmeans_eval.inertia_:.2f} | Silhouette Score: {sil_score:.4f}")

# ==========================================
# 4. VISUALISASI ELBOW & SILHOUETTE SCORE
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 5))
color = 'tab:blue'
ax1.set_xlabel('Jumlah Klaster (k)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Inertia / WCSS', color=color, fontsize=12, fontweight='bold')
ax1.plot(list(k_range), inertia_list, marker='o', linewidth=2, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Silhouette Score', color=color, fontsize=12, fontweight='bold')
ax2.plot(list(k_range), silhouette_list, marker='s', linewidth=2, linestyle='--', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Evaluasi Klaster Optimal: Elbow Method & Silhouette Score', fontsize=14, fontweight='bold', pad=15)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Simpan langsung ke folder proyek
eval_plot_path = os.path.join(BASE_DIR, 'Evaluasi_K_Optimal_Elbow_Silhouette.png')
plt.savefig(eval_plot_path, dpi=300)
plt.close()
print(f"\n[OK] Grafik evaluasi tersimpan di: {eval_plot_path}")

# ==========================================
# 5. TRAINING MODEL FINAL K-MEANS (K=4)
# ==========================================
optimal_k = 4
print(f"\nMelatih model K-Means final (K={optimal_k}) pada seluruh {len(df_rfm):,} nasabah...")
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df_rfm['Cluster'] = kmeans_final.fit_predict(rfm_scaled)

# ==========================================
# 6. KARAKTERISASI SEGMEN & MAPPING PERSONA
# ==========================================
cluster_summary = df_rfm.groupby('Cluster').agg(
    total_nasabah=('CustomerID', 'count'),
    rata_recency=('recency_days', 'mean'),
    rata_frequency=('frequency', 'mean'),
    rata_monetary=('monetary', 'mean'),
    median_monetary=('monetary', 'median')
).reset_index()

cluster_summary['persentase'] = (cluster_summary['total_nasabah'] / len(df_rfm)) * 100
# Urutkan berdasarkan rata-rata nominal belanja tertinggi
cluster_summary = cluster_summary.sort_values(by='rata_monetary', ascending=False).reset_index(drop=True)

persona_labels = [
    'Platinum VIP (High Value, Active)',
    'Gold Loyalists (Consistent Volume)',
    'Silver Mainstream (Moderate Value)',
    'Bronze Dormant (Low Value, High Inactivity)'
]
cluster_summary['Persona'] = persona_labels

# Mapping kembali nama persona ke dalam dataset nasabah
cluster_to_persona = dict(zip(cluster_summary['Cluster'], cluster_summary['Persona']))
df_rfm['Persona'] = df_rfm['Cluster'].map(cluster_to_persona)

print("\n--- RINGKASAN PROFIL KLASTER PERBANKAN ---")
print(cluster_summary[['Persona', 'total_nasabah', 'persentase', 'rata_recency', 'rata_frequency', 'rata_monetary']])

# ==========================================
# 7. VISUALISASI DISTRIBUSI SEGMEN KLASTER
# ==========================================
plt.figure(figsize=(11, 6))
palette = ['#1a365d', '#2b6cb0', '#4299e1', '#bee3f8']
ax = sns.barplot(
    data=cluster_summary,
    x='Persona',
    y='total_nasabah',
    palette=palette
)

plt.title('Distribusi Segmen Nasabah Bank Hasil K-Means Clustering', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Segmen Nasabah', fontsize=11, fontweight='bold')
plt.ylabel('Jumlah Nasabah', fontsize=11, fontweight='bold')
plt.xticks(rotation=12, ha='right')

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f"{int(height):,}\n({height/len(df_rfm)*100:.1f}%)",
                (p.get_x() + p.get_width() / 2., height / 2),
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')

plt.tight_layout()
dist_plot_path = os.path.join(BASE_DIR, 'Distribusi_Segmen_KMeans.png')
plt.savefig(dist_plot_path, dpi=300)
plt.close()
print(f"[OK] Grafik distribusi segmen tersimpan di: {dist_plot_path}")

# ==========================================
# 8. VISUALISASI SEBARAN KLASTER (SCATTER)
# ==========================================
plt.figure(figsize=(11, 7))
sample_plot = df_rfm.sample(n=10000, random_state=42)
sns.scatterplot(
    data=sample_plot,
    x='recency_days',
    y='monetary',
    hue='Persona',
    palette='tab10',
    alpha=0.6,
    s=35
)
plt.yscale('log')
plt.title('Sebaran Klaster Nasabah: Recency vs Monetary (Log-Scaled)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Recency (Hari sejak transaksi terakhir)', fontsize=11, fontweight='bold')
plt.ylabel('Monetary (Total Transaksi - Skala Log)', fontsize=11, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(title='Segmen Persona', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

scatter_plot_path = os.path.join(BASE_DIR, 'Scatter_Recency_vs_Monetary_Clusters.png')
plt.savefig(scatter_plot_path, dpi=300)
plt.close()
print(f"[OK] Grafik sebaran klaster tersimpan di: {scatter_plot_path}")

print("\nSeluruh tahapan pemodelan K-Means dan visualisasi selesai dengan sukses!")

print("\nMelatih Surrogate Classifier untuk menghitung Feature Importance...")
rf_surrogate = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# Fit model dengan data fitur transformasi dan target berupa Cluster K-Means
rf_surrogate.fit(df_transformed, df_rfm['Cluster'])

# Ambil bobot kepentingan fitur
importances = rf_surrogate.feature_importances_
feature_names = ['Recency', 'Frequency', 'Monetary']

df_fi = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=df_fi, x='Importance', y='Feature', palette='Blues_r')
plt.title('Feature Importance: Variabel Paling Berpengaruh pada Klaster', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Tingkat Kepentingan (Relative Importance)', fontsize=10, fontweight='bold')
plt.ylabel('Fitur RFM', fontsize=10, fontweight='bold')

for index, value in enumerate(df_fi['Importance']):
    plt.text(value + 0.01, index, f"{value*100:.1f}%", va='center', fontweight='bold')

plt.xlim(0, max(importances) + 0.1)
plt.tight_layout()
fi_plot_path = os.path.join(BASE_DIR, 'Feature_Importance_RFM_Clustering.png')
plt.savefig(fi_plot_path, dpi=300)
plt.close()
print(f"[OK] Grafik Feature Importance tersimpan di: {fi_plot_path}")

# ==========================================
# 10. AGREEMENT MATRIX (SQL RULE-BASED VS K-MEANS)
# ==========================================
# Replikasi logika rule-based SQL untuk perbandingan
df_rfm['r_score'] = pd.qcut(df_rfm['recency_days'], q=4, labels=[4, 3, 2, 1]).astype(int)
df_rfm['f_score'] = df_rfm['frequency'].apply(lambda x: 1 if x == 1 else (2 if x == 2 else (3 if x == 3 else 4)))
df_rfm['m_score'] = pd.qcut(df_rfm['monetary'], q=4, labels=[1, 2, 3, 4]).astype(int)

def assign_sql_segment(row):
    if row['r_score'] >= 3 and row['f_score'] >= 3 and row['m_score'] >= 3:
        return 'Champions/VIP'
    elif row['r_score'] >= 3 and row['f_score'] >= 2:
        return 'Loyal Customers'
    elif row['r_score'] >= 3 and row['f_score'] == 1:
        return 'Recent New'
    elif row['r_score'] <= 2 and row['f_score'] >= 3:
        return 'At Risk'
    elif row['r_score'] <= 2 and row['f_score'] >= 2:
        return 'Need Attention'
    else:
        return 'Hibernating'

df_rfm['SQL_Segment'] = df_rfm.apply(assign_sql_segment, axis=1)

# Buat Cross-Tabulation Matrix (Analog Confusion Matrix untuk Segmentasi)
confusion_matrix_analog = pd.crosstab(
    df_rfm['SQL_Segment'], 
    df_rfm['Persona'], 
    normalize='index'
) * 100

plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix_analog, annot=True, fmt='.1f', cmap='Blues', cbar=True)
plt.title('Agreement Matrix (%): SQL Rule-Based vs K-Means Clusters', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('K-Means Machine Learning Persona', fontsize=10, fontweight='bold')
plt.ylabel('SQL Rule-Based Segment', fontsize=10, fontweight='bold')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()

cm_plot_path = os.path.join(BASE_DIR, 'Agreement_Confusion_Matrix_Clustering.png')
plt.savefig(cm_plot_path, dpi=300)
plt.close()
print(f"[OK] Grafik Agreement/Confusion Matrix tersimpan di: {cm_plot_path}")