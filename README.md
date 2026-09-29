# 🏦 End-to-End Bank Customer Segmentation: RFM Analysis & Unsupervised Machine Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![SQL](https://img.shields.io/badge/SQL-SQLite-003B57.svg)](https://www.sqlite.org/)
[![Machine Learning](https://img.shields.io/badge/ML-K--Means%20Clustering-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Proyek analitik perbankan komprehensif yang mengintegrasikan eksplorasi data relasional tingkat lanjut (**Advanced SQL**) dan pemodelan segmentasi tanpa supervisi (**Unsupervised Machine Learning K-Means Clustering**) pada portofolio transaksi perbankan riil berskala **1.048.567 transaksi** dari **884.265 nasabah unik**.

Proyek ini membandingkan segmentasi berbasis aturan bisnis (**Rule-Based RFM Scoring**) dengan pengelompokan spasial matematis (**K-Means**), dilengkapi dengan evaluasi pemisahan klaster (*Elbow Method* & *Silhouette Analysis*), transparansi fitur (*Surrogate Feature Importance*), serta matriks silang (*Agreement Matrix*).

---

## 📌 Business Problem & Objectives

Dalam sektor perbankan ritel, strategi pemasaran seragam (*mass-marketing*) menimbulkan inefisiensi alokasi modal promosi, penurunan retensi nasabah bernilai tinggi, dan ketidakmampuan mendeteksi churn dini. 

Tujuan strategis dari proyek ini adalah:
1. **Mengkuantifikasi Metrik RFM:** Menghitung *Recency*, *Frequency*, dan *Monetary* di level nasabah menggunakan agregasi SQL berkinerja tinggi.
2. **Segmentasi Berbasis Logika Bisnis:** Mengelompokkan nasabah ke dalam 6 profil berbasis kuartil (*NTILE*) untuk kebutuhan operasional kampanye bank.
3. **Optimasi Berbasis Machine Learning:** Menerapkan K-Means Clustering dengan penanganan *skewness* distribusi perbankan untuk menemukan klaster nasabah alami.
4. **Model Explainability & Cross-Validation:** Membedah pengaruh fitur pembentuk klaster menggunakan model *Surrogate Random Forest* dan mengukur tingkat keselarasan (*Agreement Rate*) antar kedua metodologi.

---

## 🛠️ Tech Stack & Architecture

* **Database Engine:** SQLite (DB Browser for SQLite) untuk *data cleaning*, parsing ISO timestamp `D/M/YY`, dan *window functions*.
* **Programming & Libraries:** Python 3.12 (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).
* **Metodologi Analitik:** 
  * RFM Segmentation & Feature Engineering
  * Log-Transformation (`log1p`) & Z-Score Feature Scaling (`StandardScaler`)
  * K-Means Clustering with `k-means++` initialization
  * Cluster Evaluation: WCSS / Inertia (Elbow Method) & Silhouette Score
  * Surrogate Model Explainability (Random Forest Feature Importance)
  * Cross-Tabulation Matrix (Agreement Heatmap)

---

## 📊 Exploratory Data Analysis & Macro Insights (SQL)

Dari agregasi 1M+ transaksi perbankan, diperoleh indikator makro portofolio:
* **Total Transaksi:** 1.048.567 transaksi
* **Total Basis Nasabah Unik:** 884.265 nasabah
* **Rata-rata Nilai Transaksi:** ₹1.574,34 (terdistribusi dari transaksi ritel ₹0,01 hingga perputaran korporat ₹1.560.034,99)
* **Distribusi Frekuensi Transaksi:**
  * **83,76%** (740.653 nasabah) bertransaksi 1 kali (*single transactors*).
  * **16,24%** (143.612 nasabah) merupakan nasabah berulang (*repeat transactors*), dengan batas atas frekuensi mencapai 6 kali transaksi.
* **Konsentrasi Geografis:** Kota Mumbai (₹179,68 juta) dan New Delhi (₹160,70 juta) mendominasi akumulasi perputaran dana perbankan, dengan New Delhi mencatat rata-rata nominal tertinggi sebesar ₹1.892,26 per transaksi.

---

## 🔬 Unsupervised Machine Learning Pipeline

### 1. Evaluasi Penentuan Klaster Optimal (Elbow & Silhouette)
Distribusi data moneter dan frekuensi perbankan ditransformasikan menggunakan fungsi logaritmik $\ln(1+x)$ untuk meredam kemiringan ekstrem (*right-skewed*), diikuti standardisasi fitur. Evaluasi dilakukan pada rentang $k = 2$ hingga $k = 6$:

<p align="center">
  <img src="Evaluasi_K_Optimal_Elbow_Silhouette.png" width="750" alt="Evaluasi K Optimal">
</p>

* **Inersia / WCSS:** Mengalami penurunan paling signifikan hingga $k = 4$, di mana kurva mulai melandai secara bertahap (*diminishing returns*).
* **Silhouette Analysis:** Nilai $k = 4$ memberikan keseimbangan optimal antara skor kohesi spasial ($\approx 0,370$) dan kemampuan eksekusi operasional bisnis (*business actionability*).

---

### 2. Karakterisasi & Distribusi Segmen Klaster Nasabah

Model K-Means final ($k = 4$) membagi 883.660 nasabah ke dalam persona perbankan yang terukur:

<p align="center">
  <img src="Distribusi_Segmen_KMeans.png" width="700" alt="Distribusi Segmen KMeans">
</p>

| Klaster / Persona | Jumlah Nasabah | Porsi (%) | Rata-rata Recency | Rata-rata Frekuensi | Rata-rata Monetary | Karakteristik Perilaku |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Platinum VIP** | 142.236 | 16,1% | ~40 hari | ~2,1 kali | **₹5.183,75** | Kontributor profit tertinggi, frekuensi aktif, dan volume transaksi besar. |
| **Gold Loyalists** | 396.358 | 44,9% | ~45 hari | ~1,0 kali | **₹1.850,20** | Segmen penggerak transaksi harian dengan volume stabil. |
| **Silver Mainstream** | 3.650 | 0,4% | 0 hari | ~1,1 kali | **₹1.515,62** | Klaster anomali khusus nasabah yang bertransaksi tepat pada hari *cut-off*. |
| **Bronze Dormant** | 341.416 | 38,6% | ~68 hari | ~1,0 kali | **₹200,64** | Nilai transaksi mikro dan rentang inaktivasi yang panjang. |

---

### 3. Pemetaan Sebaran Spasial (Recency vs Monetary)

<p align="center">
  <img src="Scatter_Recency_vs_Monetary_Clusters.png" width="750" alt="Scatter Recency vs Monetary">
</p>

* Sumbu vertikal skala log mendemonstrasikan batas pemisah horizontal yang jelas di level $\approx 10^{2,5}$ antara kelompok bernilai kecil (*Bronze Dormant*) dan kelompok transaksi mapan (*Gold Loyalists*).
* Klaster *Platinum VIP* terbentuk melalui pemisahan pada dimensi *Frequency* (kedalaman fitur).

---

## 🔍 Model Explainability & Agreement Analysis

### 1. Surrogate Feature Importance
Untuk memastikan algoritma K-Means tidak bekerja sebagai kotak hitam (*black box*), model *Surrogate Random Forest* dilatih untuk merekonstruksi keputusan pengelompokan fitur:

<p align="center">
  <img src="Feature_Importance_RFM_Clustering.png" width="650" alt="Feature Importance">
</p>

* **Monetary (62,8%):** Menjadi pembeda utama pembentukan klaster akibat dispersi nominal transaksi yang sangat lebar.
* **Frequency (34,7%):** Pembeda kedua terbesar yang memisahkan nasabah berulang dari nasabah transaksi tunggal.
* **Recency (2,5%):** Memberikan kontribusi marjinal karena rentang observasi data historis bank yang relatif singkat (~80 hari).

---

### 2. Agreement Matrix: SQL Rule-Based vs K-Means
Matriks silang menguji keselarasan antara segmentasi berbasis kuartil manual (SQL) dan klaster machine learning (Python):

<p align="center">
  <img src="Agreement_Confusion_Matrix_Clustering.png" width="700" alt="Agreement Matrix">
</p>

* **Konsistensi Segmen Bernilai Tinggi:** Seluruh nasabah berfrekuensi tinggi dari aturan SQL (*At Risk* 100,0%, *Need Attention* 99,9%, *Loyal Customers* 99,0%, dan *Champions/VIP* 98,7%) diserap secara terpadu oleh klaster **Platinum VIP** pada K-Means.
* **Segmentasi Nasabah Transaksi Tunggal:** Nasabah *Hibernating* dan *Recent New* dipisahkan secara objektif oleh K-Means ke dalam *Bronze Dormant* (~45%--47%) dan *Gold Loyalists* (~52%--55%) berdasarkan daya beli nominal transaksi.

---

## 💼 Rekomendasi Strategis Perbankan (*Actionable Business Strategies*)

1. **Retensi & Layanan Prioritas (Platinum VIP):**
   * Alokasi *Dedicated Relationship Manager* (RM) dan jalur cepat *Priority Banking*.
   * Penawaran kartu kredit tier Platinum/World serta program *airport lounge access*.
2. **Upselling & Cross-Selling (Gold Loyalists):**
   * Otomasi autodebit tagihan rutin, asuransi bancassurance, dan produk investasi reksa dana mikro untuk meningkatkan frekuensi transaksi.
3. **Aktivasi Churn (Bronze Dormant):**
   * Kampanye reaktivasi digital berbasis insentif *cashback* atau pembebasan biaya transfer antarbank.
4. **Nurturing Onboarding (Silver Mainstream):**
   * Edukasi intensif fitur ekosistem *mobile banking* pada 30 hari pertama pasca-transaksi untuk mendorong transaksi kedua.

---

## 📂 Struktur Repositori

```text
├── rfm_customer_segmentation.sql              # Naskah kueri SQL end-to-end (EDA & agregasi RFM)
├── customer_segmentation_rfm.py               # Skrip Python pipeline K-Means, visualisasi & interpretasi
├── Evaluasi_K_Optimal_Elbow_Silhouette.png    # Grafik inersia & silhouette score
├── Distribusi_Segmen_KMeans.png               # Grafik distribusi jumlah nasabah per segmen
├── Scatter_Recency_vs_Monetary_Clusters.png   # Grafik sebaran spasial dimensi RFM
├── Feature_Importance_RFM_Clustering.png      # Grafik pembobotan fitur surrogate model
├── Agreement_Confusion_Matrix_Clustering.png  # Matriks silang SQL vs Machine Learning
├── .gitignore                                 # Pengecualian dataset mentah & file biner database
├── LICENSE                                    # Lisensi MIT
└── README.md                                  # Dokumentasi teknis & bisnis lengkap proyek
