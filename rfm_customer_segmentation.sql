-- 1. Mengambil 5 baris sampel data transaksi
SELECT * 
FROM bank_transactions 
LIMIT 5;

-- 2. Ringkasan Skala Portofolio Transaksi Bank
SELECT 
    COUNT(*) AS total_transaksi,
    COUNT(DISTINCT CustomerID) AS total_nasabah_unik,
    ROUND(AVG("TransactionAmount (INR)"), 2) AS rata_rata_nominal,
    MIN("TransactionAmount (INR)") AS nominal_terendah,
    MAX("TransactionAmount (INR)") AS nominal_tertinggi
FROM bank_transactions;

-- 3. Top 10 Nasabah Paling Aktif & Bernilai Tinggi
SELECT 
    CustomerID,
    COUNT(TransactionID) AS frekuensi_transaksi,
    ROUND(SUM("TransactionAmount (INR)"), 2) AS total_nominal,
    ROUND(AVG("TransactionAmount (INR)"), 2) AS rata_rata_per_transaksi,
    ROUND(AVG(CustAccountBalance), 2) AS rata_rata_saldo
FROM bank_transactions
GROUP BY CustomerID
ORDER BY frekuensi_transaksi DESC, total_nominal DESC
LIMIT 10;

-- 4. Distribusi Frekuensi Transaksi Nasabah
SELECT 
    frekuensi,
    COUNT(*) AS total_nasabah,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(DISTINCT CustomerID) FROM bank_transactions), 2) AS persentase
FROM (
    SELECT CustomerID, COUNT(TransactionID) AS frekuensi
    FROM bank_transactions
    GROUP BY CustomerID
)
GROUP BY frekuensi
ORDER BY frekuensi ASC;

-- 5. Analisis Persebaran Wilayah Transaksi Terbesar (Top 10 Kota)
SELECT 
    CustLocation AS kota,
    COUNT(*) AS total_transaksi,
    ROUND(SUM("TransactionAmount (INR)"), 2) AS total_nominal,
    ROUND(AVG("TransactionAmount (INR)"), 2) AS rata_rata_nominal
FROM bank_transactions
WHERE CustLocation IS NOT NULL
GROUP BY CustLocation
ORDER BY total_transaksi DESC
LIMIT 10;

-- 6. Membuat Tabel Agregasi Metrik RFM per Nasabah
CREATE TABLE customer_rfm_summary AS
WITH formatted_transactions AS (
    SELECT 
        CustomerID,
        "TransactionAmount (INR)" AS amount,
        printf('20%02d-%02d-%02d', 
            CAST(substr(TransactionDate, -2) AS INTEGER),
            CAST(substr(substr(TransactionDate, instr(TransactionDate, '/') + 1), 1, instr(substr(TransactionDate, instr(TransactionDate, '/') + 1), '/') - 1) AS INTEGER),
            CAST(substr(TransactionDate, 1, instr(TransactionDate, '/') - 1) AS INTEGER)
        ) AS transaction_date_iso
    FROM bank_transactions
    WHERE CustomerID IS NOT NULL 
      AND "TransactionAmount (INR)" > 0
)
SELECT 
    CustomerID,
    CAST(julianday('2016-10-21') - julianday(MAX(transaction_date_iso)) AS INTEGER) AS recency_days,
    COUNT(*) AS frequency,
    ROUND(SUM(amount), 2) AS monetary
FROM formatted_transactions
GROUP BY CustomerID;

SELECT *
FROM customer_rfm_summary
LIMIT 10;

-- 7. Segmentasi Nasabah Berdasarkan Skor RFM
WITH rfm_scores AS (
    SELECT 
        CustomerID,
        recency_days,
        frequency,
        monetary,
        NTILE(4) OVER (ORDER BY recency_days DESC) AS r_score,
        CASE 
            WHEN frequency = 1 THEN 1
            WHEN frequency = 2 THEN 2
            WHEN frequency = 3 THEN 3
            ELSE 4
        END AS f_score,
        NTILE(4) OVER (ORDER BY monetary ASC) AS m_score
    FROM customer_rfm_summary
),
segmented_customers AS (
    SELECT 
        CustomerID,
        recency_days,
        frequency,
        monetary,
        CASE 
            WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Champions / VIP'
            WHEN r_score >= 3 AND f_score >= 2 THEN 'Loyal Customers'
            WHEN r_score >= 3 AND f_score = 1 THEN 'Recent New Customers'
            WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk / High Value Churn'
            WHEN r_score <= 2 AND f_score >= 2 THEN 'Need Attention'
            ELSE 'Hibernating / Inactive'
        END AS segment
    FROM rfm_scores
)
SELECT 
    segment,
    COUNT(*) AS total_nasabah,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customer_rfm_summary), 2) AS persentase,
    ROUND(AVG(recency_days), 1) AS rata_rata_recency_hari,
    ROUND(AVG(frequency), 2) AS rata_rata_frekuensi,
    ROUND(AVG(monetary), 2) AS rata_rata_moneter
FROM segmented_customers
GROUP BY segment
ORDER BY rata_rata_moneter DESC;