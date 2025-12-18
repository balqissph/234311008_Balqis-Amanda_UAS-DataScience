# 📘 Judul Proyek
Prediksi Niat Pembelian Pengunjung E-commerce Menggunakan Machine Learning

## 👤 Informasi
- **Nama:** Balqis Amanda Putri Hambali  
- **Repo:** [https://github.com/balqissph/234311008_Balqis_UAS_Data-Science](https://github.com/balqissph/234311008_Balqis-Amanda_UAS-DataScience) 
- **Video:** [...]  

---

# 1. 🎯 Ringkasan Proyek
Proyek ini bertujuan untuk memprediksi purchase intention (niat pembelian) pengunjung website e-commerce berdasarkan data perilaku pengguna. Pendekatan yang digunakan mencakup proses data preparation, pemodelan machine learning, hingga evaluasi performa model.

Tahapan utama proyek meliputi:
- Melakukan eksplorasi dan pembersihan data untuk memastikan kualitas dataset.
- Menerapkan data transformation seperti encoding dan scaling agar data siap digunakan oleh model.
- Membagi dataset menjadi training, validation, dan testing set secara stratified.  
- Membangun dan membandingkan 3 model: **Baseline**, **Advanced**, **Deep Learning**  
- Melakukan evaluasi performa model menggunakan metrik klasifikasi.
- Menentukan model terbaik berdasarkan hasil evaluasi dan trade-off performa.

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
- Platform kesulitan mengidentifikasi pengunjung yang berpotensi melakukan pembelian berdasarkan perilaku kunjungan website.
- Diperlukan model machine learning yang mampu memprediksi purchase intention secara akurat menggunakan fitur-fitur perilaku pengguna seperti durasi akses, jumlah halaman, page value, dan exit rate.
- Dataset mengandung variasi nilai dan pola yang kompleks sehingga membutuhkan preprocessing dan teknik pemodelan yang tepat untuk mendapatkan performa prediksi optimal.
- Diperlukan model machine learning yang mampu memprediksi purchase intention secara akurat menggunakan fitur-fitur perilaku pengguna seperti durasi akses, jumlah halaman, page value, dan exit rate.
  

**Goals:**  
- Membangun model machine learning yang mampu memprediksi purchase intention pengunjung website dengan tingkat akurasi minimal 80%.
- Menganalisis dan membandingkan performa tiga pendekatan model (baseline, model ensemble, dan deep learning) menggunakan metrik evaluasi seperti accuracy, precision, recall, dan F1-score.
- Menentukan model terbaik yang mampu mengenali pola perilaku pengguna secara konsisten berdasarkan hasil evaluasi.
- Menghasilkan proses pengolahan data dan pelatihan model yang reproducible, sehingga dapat dijalankan ulang tanpa error pada lingkungan pengembangan yang sama.
  

---
## 📁 Struktur Folder
```
project/
│
├── data/                   # Dataset (tidak di-commit, download manual)
│
├── notebooks/              # Jupyter notebooks
│   └── UAS_Data_Science.ipynb
│
├── src/                    # Source code
│   
├── models/                 # Saved models
│   ├── logistic_regression_model.pkl
│   ├── mlp_model.h5
│   ├── mlp_model.keras
│   └── random_forest_model.pkl
│
├── images/                 # Visualizations
│   ├── Confusion Matrix Model DL.png
│   ├── Confusion Matrix Model LR.png
│   ├── Confusion Matrix Model RF.png
│   ├── Contoh Hasil Prediksi - DL.png
│   ├── Featur Importance - RF.png
│   ├── History Training - DL.png
│   ├── Perbandingan Performa Ketiga Model.png
│   ├── Training & Validation Accuracy per Epoch - DL.png
│   ├── Training & Validation Loss per Epoch - DL.png
│   ├── V1 - Distribusi Label Target.png
│   ├── V2 - Bloxpot Fitur Durasi.png
│   └── V3 - Heatmap Korelasi Fitur Numerik.png
│
├── requirements.txt        # Dependencies
├── .gitignore
├── LICENSE
├── Ceklist Submit.md
├── requirements.txt
├── Laporan Proyek UAS Machine Learning.pdf
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber:** UC Irvine (Machine Learning Repository) https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset   
- **Jumlah Data:** 12.330 baris x 18 kolom  
- **Tipe:** CSV (Data Tabular)  

### Fitur Utama
| Fitur | Deskripsi |
|------|-----------|
| Administrative | Jumlah halaman administratif yang dikunjungi pengunjung. |
| Administrative_Duration | Total durasi (dalam detik) yang dihabiskan pada halaman administratif. |
| Informational | Jumlah halaman informasi yang dikunjungi pengunjung. |
| Informational_Duration | Total durasi waktu yang dihabiskan pada halaman informasi. |
| ProductRelated | Jumlah halaman produk yang dikunjungi pengunjung. |
| ProductRelated_Duration | Total durasi waktu yang dihabiskan pada halaman produk. |
| BounceRates | Persentase pengunjung yang keluar setelah hanya melihat satu halaman. |
| ExitRates | Persentase pengunjung yang keluar dari situs setelah mengunjungi halaman tertentu. |
| PageValues | Nilai estimasi kontribusi halaman terhadap terjadinya konversi pembelian. |
| SpecialDay | Tingkat kedekatan hari kunjungan dengan hari-hari khusus atau promosi. |
| Month | Bulan terjadinya sesi kunjungan pengguna. |
| OperatingSystems | Sistem operasi yang digunakan oleh pengunjung website. |
| Browser | Jenis browser yang digunakan oleh pengunjung. |
| Region | Wilayah asal pengunjung website. |
| TrafficType | Jenis sumber trafik yang mengarahkan pengunjung ke website. |
| VisitorType | Jenis pengunjung berdasarkan riwayat kunjungan (baru atau kembali). |
| Weekend | Menunjukkan apakah kunjungan terjadi pada akhir pekan. |
| Revenue | Label target yang menunjukkan apakah pengunjung melakukan pembelian. |

---

# 4. 🔧 Data Preparation
- Cleaning: pengecekan missing values, penghapusan duplikasi, analisis outliers
- Transformation: encoding biner (0/1) dan scaling selektif dengan RobustScaler
- Splitting: train / validation / test menggunakan stratified split  

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** Logistic Regression  
- **Model 2 – Advanced ML:** Random Forest  
- **Model 3 – Deep Learning:** Multilayer Perceptron (MLP)  

---

# 6. 🧪 Evaluation
**Metrik:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
### Hasil Singkat
| Model | Accuracy | Catatan |
|------|----------|---------|
| Baseline (Logistic Regression) | 0.892 | Model sederhana dengan performa cukup baik sebagai pembanding awal |
| Advanced (Random Forest) | 0.908 | Performa terbaik, mampu menangkap pola non-linear dengan lebih baik |
| Deep Learning (MLP) | 0.905 | Performa tinggi namun membutuhkan waktu training lebih lama |

---

# 7. 🏁 Kesimpulan
- Model terbaik: Random Forest
- Alasan: Memberikan performa paling stabil dengan akurasi dan F1-score tertinggi.
- Insight: Perilaku interaksi pengguna seperti durasi dan halaman produk sangat berpengaruh terhadap purchase intention.  

---

# 8. 🔮 Future Work
- [x] Tambah data  
- [x] Tuning model  
- [x] Coba arsitektur DL lain  
- [x] Deployment  

---

# 9. 🔁 Reproducibility
### Environment
- **Python Version:** 3.12.12  
- **Platform:** Google Colab / Local Machine  
- **Hardware:** CPU  

### Library Versions
- numpy==2.0.2  
- pandas==2.2.2  
- scikit-learn==1.6.1  
- matplotlib==3.10.0  
- seaborn==0.13.2  
- tensorflow==2.19.0  
- keras==3.10.0  
