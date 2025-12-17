**INFORMASI PROYEK**

**Judul Proyek :**

**Prediksi Niat Pembelian Pengunjung E-commerce Menggunakan
Machine Learning**

  -------------------------------------------------------------------------------------------
  **Nama Mahasiswa**   :   BALQIS AMANDA PUTRI HAMBALI
  -------------------- --- ------------------------------------------------------------------
  **NIM**              :   234311008

  **Program Studi**    :   TEKNOLOGI REKAYASA PERANGKAT LUNAK

  **Mata Kuliah**      :   DATA SCIENCE

  **Dosen Pengampu**   :   GUS NANANG SYAIFUDDIIN

  **Tahun Akademik**   :   2025/5

  **Link GitHub        :   <https://github.com/balqissph/234311008_Balqis_UAS_Data-Science>
  Repository**             

  **Link Video         :   \[URL Repository\]
  Pembahasan**             
  -------------------------------------------------------------------------------------------

1.  **LEARNING OUTCOMES**

Pada proyek ini, mahasiswa diharapkan dapat:

1.  Memahami konteks masalah dan merumuskan problem statement secara
    > jelas

2.  Melakukan analisis dan eksplorasi data (EDA) secara komprehensif
    > (**OPSIONAL**)

3.  Melakukan data preparation yang sesuai dengan karakteristik dataset

4.  Mengembangkan tiga model machine learning yang terdiri dari
    > (**WAJIB**):

    -   Model baseline

    -   Model machine learning / advanced

    -   Model deep learning (**WAJIB**)

5.  Menggunakan metrik evaluasi yang relevan dengan jenis tugas ML

6.  Melaporkan hasil eksperimen secara ilmiah dan sistematis

7.  Mengunggah seluruh kode proyek ke GitHub (**WAJIB**)

8.  Menerapkan prinsip software engineering dalam pengembangan proyek

```{=html}
<!-- -->
```
2.  **PROJECT OVERVIEW**

    1.  **Latar Belakang**

> Pertumbuhan aktivitas belanja online menyebabkan data perilaku
> pengunjung website menjadi semakin penting untuk dianalisis. Namun,
> tidak semua pengunjung yang mengakses platform e-commerce akan
> melakukan pembelian. Banyak pengunjung hanya melihat produk tanpa
> melanjutkan transaksi, sehingga menimbulkan kesenjangan antara
> tingginya jumlah kunjungan dan rendahnya tingkat konversi. Kondisi ini
> menyulitkan platform dalam memahami faktor-faktor yang memengaruhi
> keputusan pembelian pengguna.
>
> Permasalahan tersebut umum ditemukan dalam analisis perilaku pengguna,
> di mana data interaksi seperti jumlah halaman yang dikunjungi, durasi
> kunjungan, nilai halaman, dan tingkat keluar sering kali belum
> dimanfaatkan secara optimal. Tanpa pendekatan analisis berbasis data,
> platform akan kesulitan mengidentifikasi pengunjung yang memiliki
> potensi untuk melakukan pembelian.
>
> Penerapan machine learning menjadi solusi yang relevan karena mampu
> mempelajari pola perilaku pengguna dan memprediksi kemungkinan
> terjadinya pembelian (purchase intention). Pendekatan ini dapat
> membantu meningkatkan efektivitas strategi pemasaran, personalisasi
> layanan, serta peluang konversi. Beberapa penelitian telah menunjukkan
> bahwa machine learning efektif dalam memprediksi niat pembelian
> berdasarkan data perilaku pengunjung. Misalnya, studi oleh (Cucu Ika
> Agustyaningrum dkk., 2021) membandingkan berbagai algoritma machine
> learning dan deep neural network dalam menganalisis niat belanja
> online, dan hasilnya menunjukkan bahwa pola interaksi pengguna dapat
> digunakan sebagai fitur prediktif yang kuat. Selain itu, penelitian
> (Sintia dkk., 2023) menegaskan bahwa faktor perilaku dan persepsi
> pengguna berpengaruh signifikan terhadap keputusan pembelian.
>
> Berdasarkan hal tersebut, proyek ini memiliki nilai penting baik
> secara praktis maupun akademik. Secara praktis, model prediksi dapat
> membantu platform memahami perilaku pengunjung dan meningkatkan
> strategi pemasaran. Secara akademik, proyek ini memperkuat penelitian
> mengenai penerapan machine learning dalam memodelkan perilaku konsumen
> berbasis data.
>
> **Referensi :**
>
> Cucu Ika Agustyaningrum, Haris, M., Aryanti, R., & Misriati, T.
> (2021). Online Shopper Intention Analysis Using Conventional Machine
> Learning And Deep Neural Network Classification Algorithm. *Jurnal
> Penelitian Pos dan Informatika*, *11*(1), 89--100.
> https://doi.org/10.17933/jppi.v11i1.341
>
> Sintia, L., Siagian, Y. M., & Kurniawati, K. (2023). The Determinants
> of Purchase Intention in Social Commerce. *Jurnal Manajemen Bisnis*,
> *14*(1), 214--237. https://doi.org/10.18196/mb.v14i1.15754

3.  **BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING**

```{=html}
<!-- -->
```
1.  **Problem Statements**

```{=html}
<!-- -->
```
1.  Platform kesulitan mengidentifikasi pengunjung yang berpotensi
    melakukan pembelian berdasarkan perilaku kunjungan website.

2.  Diperlukan model machine learning yang mampu memprediksi purchase
    intention secara akurat menggunakan fitur-fitur perilaku pengguna
    seperti durasi akses, jumlah halaman, page value, dan exit rate.

3.  Dataset mengandung variasi nilai dan pola yang kompleks sehingga
    membutuhkan preprocessing dan teknik pemodelan yang tepat untuk
    mendapatkan performa prediksi optimal.

4.  Diperlukan model machine learning yang mampu memprediksi purchase
    intention secara akurat menggunakan fitur-fitur perilaku pengguna
    seperti durasi akses, jumlah halaman, page value, dan exit rate.

```{=html}
<!-- -->
```
2.  **Goals**

```{=html}
<!-- -->
```
1.  Membangun model machine learning yang mampu memprediksi purchase
    intention pengunjung website dengan tingkat akurasi minimal 80%.

2.  Menganalisis dan membandingkan performa tiga pendekatan model
    (baseline, model ensemble, dan deep learning) menggunakan metrik
    evaluasi seperti accuracy, precision, recall, dan F1-score.

3.  Menentukan model terbaik yang mampu mengenali pola perilaku pengguna
    secara konsisten berdasarkan hasil evaluasi.

4.  Menghasilkan proses pengolahan data dan pelatihan model yang
    reproducible, sehingga dapat dijalankan ulang tanpa error pada
    lingkungan pengembangan yang sama.

```{=html}
<!-- -->
```
3.  **Solution Approach**

> **Model 1 -- Baseline Model**
>
> Model yang Dipilih: **Logistic Regression**
>
> Sebagai model baseline, Logistic Regression dipilih karena merupakan
> algoritma klasifikasi sederhana yang banyak digunakan untuk memecahkan
> permasalahan klasifikasi biner, termasuk prediksi purchase intention.
> Model ini mampu memberikan gambaran awal seberapa baik fitur-fitur
> pada dataset, seperti durasi kunjungan, jumlah halaman yang diakses,
> bounce rate, exit rate, dan page value dalam memprediksi apakah
> seorang pengunjung akan melakukan pembelian atau tidak.
>
> Logistic Regression juga mudah diinterpretasikan dan memiliki proses
> pelatihan yang cepat, sehingga ideal sebagai titik awal perbandingan
> dengan model lain yang lebih kompleks. Dengan menggunakan model ini
> sebagai baseline, performanya dapat menjadi acuan untuk menilai apakah
> model-model lanjutan (ensemble maupun deep learning) benar-benar
> meningkatkan akurasi dan ketepatan prediksi.
>
> **Model 2 -- Advanced / ML Model**
>
> Model yang Dipilih: **Random Forest**
>
> Random Forest dipilih sebagai model advanced karena merupakan
> algoritma ensemble learning berbasis bagging yang mampu mengatasi
> kompleksitas pola pada data perilaku pengguna e-commerce. Berbeda
> dengan model baseline seperti Logistic Regression yang bersifat
> linear, Random Forest mampu menangkap hubungan non-linear antar fitur
> seperti jumlah halaman yang dikunjungi, durasi akses, bounce rate,
> exit rate, dan page value, sehingga lebih efektif dalam memprediksi
> purchase intention.
>
> Keunggulan Random Forest juga terletak pada kemampuannya mengurangi
> overfitting melalui penggunaan banyak pohon keputusan (decision trees)
> yang digabungkan untuk menghasilkan prediksi yang lebih stabil. Model
> ini dikenal memiliki performa yang kuat pada dataset tabular seperti
> dataset yang digunakan dalam proyek ini. Selain itu, Random Forest
> dapat memberikan informasi penting mengenai kontribusi masing-masing
> fitur (feature importance), sehingga membantu dalam memahami faktor
> perilaku pengguna yang berperan dalam keputusan pembelian.
>
> **Model 3 -- Deep Learning Model (WAJIB)**
>
> Model yang Dipilih: **Multilayer Perceptron (MLP) / Neural Network**
>
> Karena dataset Online Shoppers Purchasing Intention merupakan **data
> tabular** dengan fitur numerik dan kategorikal, model deep learning
> yang paling tepat digunakan adalah Multilayer Perceptron (MLP). MLP
> adalah arsitektur jaringan saraf feed-forward yang efektif untuk tugas
> klasifikasi biner dan mampu mempelajari hubungan non-linear yang tidak
> dapat ditangkap dengan baik oleh model linear seperti Logistic
> Regression.
>
> Model MLP dalam proyek ini akan dibangun dengan minimal 2 hidden
> layers, mengikuti ketentuan UAS, sehingga mampu mengekstraksi
> representasi fitur yang lebih kompleks. Dengan aktivasi non-linear
> seperti ReLU dan regularisasi seperti dropout atau batch
> normalization, MLP dapat mempelajari struktur pola dari fitur perilaku
> pengguna seperti durasi akses, jumlah halaman, page value, bounce
> rate, dan exit rate secara lebih mendalam.
>
> MLP dipilih karena:

1.  Cocok untuk data tabular dengan kombinasi fitur numerik dan
    kategorikal.

2.  Mampu menangkap hubungan non-linear, yang sering muncul pada pola
    interaksi pengunjung website.

3.  Lebih fleksibel dibanding model tradisional, dan mampu meningkatkan
    performa bila dioptimasi dengan baik.

4.  Sesuai dengan ketentuan UAS yang mewajibkan deep learning minimal 10
    epochs, sehingga dapat dilakukan tracking terhadap training loss,
    validation loss, accuracy, serta hasil prediksi pada test set.

5.  Lebih mudah diimplementasikan dibanding model deep learning berbasis
    gambar atau teks, karena struktur datanya sudah dalam format
    numerik.

```{=html}
<!-- -->
```
4.  **DATA UNDERSTANDING**

    1.  **Informasi Dataset**

> **Sumber Dataset:**
>
> UC Irvine (Machine Learning Repository :
>
> <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset>
>
> **Deskripsi Dataset:**

-   Jumlah baris (rows): 12.330

-   Jumlah kolom (columns/features): 17 fitur dan 1 target

-   Tipe data: Tabular (numerik dan kategorikal)

-   Ukuran dataset: 1.1 MB

-   Format file: CSV

    1.  **Deskripsi Fitur**

  ---------------------------------------------------------------------------
  **Nama Fitur**     **Tipe Data** **Deskripsi**               **Contoh
                                                               Nilai**
  ------------------ ------------- --------------------------- --------------
  Administrative     Integer       Jumlah halaman              0, 1, 3
                                   administratif yang          
                                   dikunjungi pengunjung.      

  Administrative\_   Float         Total durasi (detik) yang   0.0, 5.5, 20.3
  Duration                         dihabiskan pada halaman     
                                   administratif.              

  Informational      Integer       Jumlah halaman informasi    0, 2, 4
                                   yang dikunjungi.            

  Informational\_    Float         Durasi waktu pada halaman   0.0, 10.2,
  Duration                         informasi.                  35.1

  ProductRelated     Integer       Jumlah halaman produk yang  1, 10, 50
                                   dikunjungi.                 

  ProductRelated\_   Float         Durasi waktu yang           0.0, 150.5,
  Duration                         dihabiskan pada halaman     600.2
                                   produk.                     

  BounceRates        Float         Persentase pengunjung yang  0.02, 0.20
                                   keluar setelah melihat satu 
                                   halaman.                    

  ExitRates          Float         Persentase pengunjung       0.05, 0.30
                                   keluar dari situs setelah   
                                   halaman tertentu.           

  PageValues         Float         Nilai estimasi konversi     0.0, 5.7, 20.3
                                   dari halaman yang           
                                   dikunjungi.                 

  SpecialDay         Float         Nilai kedekatan hari        0.0, 0.4, 1.0
                                   kunjungan dengan hari-hari  
                                   khusus (misal Hari Besar).  

  Month              Categorical   Bulan terjadinya sesi       \"Feb\",
                                   kunjungan.                  \"Jul\",
                                                               \"Dec\"

  OperatingSystems   Integer       Sistem operasi yang         1, 2, 3, 4
                                   digunakan pengunjung.       

  Browser            Integer       Browser yang digunakan      1, 2, 5, 7
                                   pengunjung.                 

  Region             Integer       Wilayah asal pengunjung.    1, 3, 5, 9

  TrafficType        Integer       Jenis sumber trafik         1, 2, 10, 20
                                   pengunjung.                 

  VisitorType        Categorical   Jenis pengunjung: baru atau \"New
                                   kembali.                    Visitor\",
                                                               \"Returning
                                                               Visitor\"

  Weekend            Boolean       Apakah kunjungan terjadi    True / False
                                   pada akhir pekan.           

  Revenue            Boolean       Label target: apakah        True / False
                     (Target)      pengunjung melakukan        
                                   pembelian.                  
  ---------------------------------------------------------------------------

2.  **Kondisi Data**

-   **Missing Values:** Tidak Ada

-   **Duplicate Data:** Ada, 125 baris

-   **Outliers:** Ada, pada fitur :

    -   Administrative: 404 outliners

    -   Administrative_Duration: 1172 outliners

    -   Administrative: 404 outliers

    -   Administrative_Duration: 1172 outliers

    -   Informational: 2631 outliers

    -   Informational_Duration: 2405 outliers

    -   ProductRelated: 987 outliers

    -   ProductRelated_Duration: 961 outliers

    -   BounceRates: 1551 outliers

    -   ExitRates: 1099 outliers

    -   PageValues: 2730 outliers

    -   SpecialDay: 1251 outliers

    -   OperatingSystems: 111 outliers

    -   Browser: 4369 outliers

    -   Region: 511 outliers

    -   TrafficType: 2101 outliers

-   **Imbalanced Data:** Ada, label target Revenue bersifat imbalanced.

-   **Noise:** Ada, beberapa fitur perilaku seperti bounce rate dan exit
    > rate dapat menunjukkan variasi ekstrem yang tidak selalu
    > merepresentasikan perilaku nyata, akibat perbedaan perangkat,
    > koneksi internet, atau sesi singkat. Namun jumlah noise tidak
    > signifikan dan masih dapat ditangani oleh model ML.

-   **Data Quality Issues:** Secara umum dataset memiliki kualitas baik,
    > namun terdapat karakteristik khusus :

    -   Fitur Month berisi singkatan bulan (Jan, Feb, Mar, ...) sehingga
        perlu one-hot encoding.

    -   Variasi nilai pada PageValues cukup tajam, sehingga scaling
        diperlukan sebelum training model tertentu.

    1.  **Exploratory Data Analysis (EDA)**

> **Visualisasi 1: Distribusi Label Target (Revenue)**
>
> ![](media/image1.png){width="3.9006594488188977in"
> height="2.5635378390201224in"}
>
> **Insight:**\
> Grafik menunjukkan bahwa dataset memiliki distribusi kelas yang sangat
> tidak seimbang (imbalanced). Sebagian besar data berada pada kelas
> False, yaitu pengunjung yang tidak melakukan pembelian. Jumlahnya
> tampak mencapai lebih dari 10.000 sesi, sementara kelas True hanya
> sekitar 2.000 sesi. Ketidakseimbangan ini menunjukkan bahwa hanya
> sebagian kecil pengunjung website yang benar-benar melakukan
> transaksi.
>
> **Visualisasi 2: Boxplot Fitur Durasi & Halaman (Outlier Detection)**
>
> ![](media/image2.png){width="5.272469378827647in"
> height="2.9343449256342957in"}
>
> **Insight:**\
> Boxplot menunjukkan adanya jumlah outliers yang sangat besar pada
> ketiga fitur durasi, terutama pada fitur ProductRelated_Duration, di
> mana terdapat banyak nilai ekstrem hingga lebih dari 60.000 detik. Hal
> ini menggambarkan bahwa perilaku pengunjung sangat bervariasi, ada
> yang hanya mengakses sebentar, tetapi ada pula yang menghabiskan waktu
> sangat lama saat melihat halaman produk. Outliers seperti ini umum
> terjadi pada data perilaku pengguna online dan tidak selalu
> menunjukkan kualitas data yang buruk. Namun, kondisi ini perlu
> diperhatikan karena model yang sensitif terhadap outliers seperti
> Logistic Regression atau KNN bisa terpengaruh. Pada preprocessing
> nanti, penggunaan RobustScaler menjadi opsi yang tepat karena lebih
> stabil terhadap outliers.
>
> **Visualisasi 3: Heatmap Korelasi antar Fitur Numerik**
>
> ![](media/image3.png){width="4.537377515310586in"
> height="4.052951662292213in"}
>
> **Insight:**\
> Heatmap korelasi menggambarkan pola hubungan antar fitur numerik.
> Terdapat korelasi positif yang kuat antara fitur ProductRelated dan
> ProductRelated_Duration, yang wajar karena semakin banyak halaman
> produk yang dikunjungi, semakin lama waktu yang dihabiskan pengguna.
> Selain itu, fitur PageValues menunjukkan korelasi positif terhadap
> beberapa fitur perilaku, yang menandakan bahwa sesi dengan nilai
> halaman yang tinggi cenderung lebih berkaitan dengan aktivitas
> mendekati transaksi. Sebaliknya, fitur seperti BounceRates dan
> ExitRates berkorelasi negatif terhadap beberapa fitur lain,
> menunjukkan bahwa pengguna yang lebih cepat meninggalkan halaman
> umumnya tidak melakukan interaksi mendalam. Heatmap ini membantu
> mengidentifikasi fitur-fitur yang memiliki potensi sebagai prediktor
> kuat dalam model klasifikasi.

5.  **DATA PREPARATION**

```{=html}
<!-- -->
```
1.  **Data Cleaning**

> **Aktivitas:**

1.  **Handling Missing Values**

> Pengecekan missing values menggunakan df.isnull().sum().

-   Hasil: 0 missing values

-   Keputusan: Tidak dilakukan imputasi karena seluruh data lengkap.

2.  **Removing duplicates**

> Pengecekan duplikasi menggunakan df.duplicated().sum().

-   Hasil: 125 row

-   Keputusan: Duplikasi dihapus menggunakan df.drop_duplicates()

-   Alasan: Duplikasi dapat menyebabkan bias pada model

3.  **Handling Outliers**

> Outliers dianalisis menggunakan metode Interquartile Range (IQR).

-   Hasil: Hampir semua fitur memiliki outliners.

-   Keputusan**:** Outliers tidak dihapus*.*

-   Alasan: Menghapus outliers berpotensi menghilangkan informasi
    penting terkait perilaku ekstrem pengguna.

4.  **Data Type Conversion**

> Dataset memiliki beberapa fitur dengan tipe data kategorikal dan
> boolean yang perlu dikonversi agar dapat diproses oleh algoritma
> machine learning.

-   Keputusan :

    -   Fitur kategorikal tertentu akan dikonversi menggunakann One-Hot
        Encoding.

    -   Fitur Boolean dan variabel target akan dikonversi menjadi nilai
        numerik 0 dan 1.

-   Alasan:

    -   Seluruh algoritma yang digunakan membutuhkan input numerik.

    -   Transformasi ini memudahkan proses training, evaluasi, dan
        perhitungan metrik klasifikasi seperti accuracy, precision,
        recall, dan F1-score.

2.  **Feature Engineering**

> **Aktivitas:**

-   **Creating New Features**

> Pada tahap ini, **tidak** dilakukan pembuatan fitur baru (feature
> creation). Karena, dataset sudah menyediakan fitur perilaku pengguna
> yang cukup representatif..

-   **Feature Extraction**

> Feature extraction **tidak** dilakukan karena dataset berbentuk data
> tabular terstruktur.

-   **Feature Selection**

> Feature selection dilakukan secara **implicit** melalui proses
> pemodelan.

-   **Dimensionality Reduction**

> Teknik dimensionality reduction seperti PCA **tidak** diterapkan pada
> penelitian ini. Karena, jumlah fitur relatif kecil dan masih dapat
> ditangani oleh model secara efisien.

3.  **Data Transformation**

> Tahap data transformation dilakukan untuk mengubah data mentah menjadi
> format yang sesuai agar dapat diproses secara optimal oleh algoritma
> machine learning. Karena dataset yang digunakan merupakan **data
> tabular**, maka proses transformasi difokuskan pada encoding dan
> scaling.

1.  **Encoding**

> Fitur kategorikal pada dataset ditransformasikan menggunakan **One-Hot
> Encoding**, sedangkan fitur bertipe boolean dan variabel target diubah
> ke bentuk numerik biner (0 dan 1).

2.  **Scaling**

> Fitur numerik dilakukan proses scaling menggunakan **RobustScaler**.
> Proses scaling hanya diterapkan pada fitur tertentu, karena pada
> beberapa fitur numerik tidak dilakukan scaling karena merupakan data
> distrik atau identifier numeirk yang tidak merepresentasikan skala
> kontinu.

4.  **Data Splitting**

> Pembagian dilakukan menggunakan stratified split untuk menjaga
> proporsi kelas target (Revenue) tetap konsisten pada setiap subset
> data, mengingat dataset bersifat tidak seimbang.
>
> Proporsi pembagian data adalah:

-   Training set: 80%

-   Validation set: 10%

-   Test set: 10%

-   nilai random state = 42 untuk reproducibility

> Alasan penggunaan strategi ini :

-   Stratified split menjaga distribusi kelas target tetap konsisten
    pada data training, validation, dan test.

-   Validation set digunakan untuk memantau performa model selama proses
    pelatihan dan mencegah overfitting.

-   Penggunaan random state memastikan konsistensi hasil eksperimen dan
    memudahkan proses evaluasi ulang.

5.  **Data Balancing**

> Berdasarkan distribusi variabel target (*Revenue*), dataset
> menunjukkan ketidakseimbangan kelas antara pengunjung yang melakukan
> pembelian dan yang tidak. Namun, pada penelitian ini **tidak dilakukan
> proses data balancing** karena distribusi tersebut mencerminkan
> kondisi nyata perilaku pengunjung website.

6.  **Ringkasan Data Preparation**

```{=html}
<!-- -->
```
1.  **Data Cleaning**

-   **Apa yang dilakukan :** mengecek mising values, hapus data duplikat
    dan menganlisis outliners.

-   **Mengapa penting :** data bersih mencegah bias, kesalahan pelatihan
    model, dan hasil evaluasi yang menyesatkan.

-   **Bagaimana implementasinya :** Dataset dipastikan tidak memiliki
    missing values, sebanyak 125 data duplikat dihapus, dan outliers
    dipertahankan karena dianggap merepresentasikan perilaku nyata
    pengguna.

2.  **Data Transformation -- Encoding**

-   **Apa yang dilakukan :** melakukan encoding pada fitur bertipe
    boolean dan kategorikal agar dapat diproses oleh algoritma machine
    learning.

-   **Mengapa penting :** algoritma machine learning memerlukan data
    numerik sehingga fitur kategorikal dan boolean perlu
    ditransformasikan ke bentuk angka.

-   **Bagaimana implementasinya :** fitur Weekend dikonversi menjadi
    nilai biner 0 dan 1, sedangkan fitur VisitorType dan Month
    diencoding ke bentuk biner 0/1, sementara fitur kode numerik lainnya
    tidak diubah.

3.  **Data Transformation -- Scalling**

-   **Apa yang dilakukan :** melakukan scaling secara selektif pada
    fitur numerik tertentu.

-   **Mengapa penting :** perbedaan skala yang besar dan keberadaan
    outliers dapat memengaruhi kinerja model, terutama model yang
    sensitif terhadap skala.

-   **Bagaimana implementasinya :** scaling dilakukan menggunakan
    RobustScaler pada fitur durasi dan metrik interaksi pengguna,
    sedangkan fitur numerik diskrit dan kode kategori tidak dilakukan
    scaling untuk menjaga interpretabilitas data.

4.  **Data Splitting**

-   **Apa yang dilakukan :** membagi dataset menjadi data training,
    validation, dan testing.

-   **Mengapa penting :** pembagian data diperlukan untuk melatih model,
    melakukan validasi selama training, serta menguji kemampuan
    generalisasi model pada data yang belum pernah dilihat.

-   **Bagaimana implementasinya :** dataset dibagi menggunakan
    stratified split agar distribusi kelas target tetap seimbang, dengan
    proporsi training, validation, dan test, serta menggunakan random
    state untuk menjaga konsistensi hasil.

6.  **MODELING**

    1.  **Model 1 --- Baseline Model**

```{=html}
<!-- -->
```
1.  **Deskripsi Model**

> **Nama Model:**  Logistic Regression
>
> **Teori Singkat:**
>
> Logistic Regression merupakan algoritma klasifikasi yang digunakan
> untuk memprediksi probabilitas suatu kejadian pada masalah klasifikasi
> biner. Model ini bekerja dengan mengombinasikan fitur input secara
> linear, kemudian menerapkan fungsi sigmoid untuk menghasilkan nilai
> probabilitas antara 0 dan 1. Nilai probabilitas tersebut selanjutnya
> dikonversi menjadi kelas prediksi berdasarkan ambang batas tertentu,
> umumnya 0.5.
>
> **Alasan Pemilihan:**

-   Model memiliki struktur sederhana dan mudah diinterpretasikan.

-   Sering digunakan sebagai baseline pada masalah klasifikasi biner.

-   Memberikan gambaran performa awal sebelum menerapkan model yang
    lebih kompleks.

2.  **Hyperparameter**

> **Parameter yang digunakan:**

-   **C (regularization):** 1.0

-   **solver:** \'lbfgs**\'**

-   **max_iter:** 1000

3.  **Implementasi (Ringkas)**

> #@title Baseline Model : LOGISTIC REGRESSION
>
> from sklearn.linear_model import LogisticRegression
>
> model_baseline = LogisticRegression(
>
>     C=1.0,
>
>     solver=\'lbfgs\',
>
>     max_iter=1000,
>
>     random_state=42
>
> )
>
> model_baseline.fit(X_train, y_train)
>
> y_pred_baseline = model_baseline.predict(X_test)

4.  **Hasil Awal**

> Accuracy: 0.8918918918918919
>
> Classification Report:
>
> precision recall f1-score support
>
> 0 0.90 0.98 0.94 1030
>
> 1 0.79 0.42 0.55 191
>
> accuracy 0.89 1221
>
> macro avg 0.84 0.70 0.74 1221
>
> weighted avg 0.88 0.89 0.88 1221
>
> Confusion Matrix:
>
> \[\[1008 22\]
>
> \[ 110 81\]\]

1.  **Model 2 --- ML / Advanced Model**

```{=html}
<!-- -->
```
1.  **Deskripsi Model**

> **Nama Model:** Random Forest 
>
> **Teori Singkat:**
>
> Random Forest merupakan algoritma *ensemble learning* yang membangun
> banyak pohon keputusan (*decision tree*) selama proses pelatihan.
> Setiap pohon dilatih menggunakan subset data dan fitur yang dipilih
> secara acak, kemudian hasil prediksi dari seluruh pohon digabungkan
> melalui mekanisme voting untuk menentukan kelas akhir. Pendekatan ini
> membantu mengurangi overfitting yang umum terjadi pada single decision
> tree.
>
> **Alasan Pemilihan:**

-   Mampu menangani data tabular dengan baik dan memodelkan hubungan
    non-linear antar fitur.

-   Relatif robust terhadap outliers dan noise pada data.

-   Umumnya memberikan performa yang lebih baik dibandingkan model
    baseline seperti Logistic Regression.

> **Keunggulan:**

-   Mampu menangkap pola non-linear yang kompleks.

-   Relatif robust terhadap outliers dan noise.

-   Tidak memerlukan scaling fitur secara menyeluruh.

-   Memiliki performa yang baik pada data tabular.

> **Kelemahan:**

-   Interpretabilitas lebih rendah dibandingkan model linear.

-   Waktu pelatihan lebih lama dibandingkan baseline model.

-   Ukuran model cenderung besar karena terdiri dari banyak pohon.

2.  **Hyperparameter**

> **Parameter yang digunakan:**

-   **n_estimators:** 100

-   **max_depth:** None

-   **min_samples_split:** 2

-   **random_state:** 42

3.  **Implementasi (Ringkas)**

> #@title ML / Advanced Model : RANDOM FOREST
>
> from sklearn.ensemble import RandomForestClassifier
>
> \# Inisialisasi model Random Forest
>
> model_rf = RandomForestClassifier(
>
>     n_estimators=100,
>
>     max_depth=None,
>
>     min_samples_split=2,
>
>     random_state=42
>
> )
>
> \# Training model
>
> model_rf.fit(X_train, y_train)
>
> \# Prediksi pada data test
>
> y_pred_rf = model_rf.predict(X_test)

4.  **Hasil Model**

> Accuracy Random Forest: 0.9082719082719083
>
> Classification Report:
>
> precision recall f1-score support
>
> 0 0.93 0.97 0.95 1030
>
> 1 0.77 0.60 0.67 191
>
> accuracy 0.91 1221
>
> macro avg 0.85 0.78 0.81 1221
>
> weighted avg 0.90 0.91 0.90 1221
>
> Confusion Matrix:
>
> \[\[995 35\]
>
> \[ 77 114\]\]

1.  **Model 3 --- Deep Learning Model (WAJIB)**

```{=html}
<!-- -->
```
1.  **Deskripsi Model**

> **Nama Model:** Multilayer Perceptron (MLP)
>
> \*\* (Centang) Jenis Deep Learning: \*\*
>
> Multilayer Perceptron (MLP) - untuk tabular
>
> Convolutional Neural Network (CNN) - untuk image
>
> Recurrent Neural Network (LSTM/GRU) - untuk sequential/text
>
> Transfer Learning - untuk image
>
>  Transformer-based - untuk NLP
>
>  Autoencoder - untuk unsupervised
>
> Neural Collaborative Filtering - untuk recommender
>
> **Alasan Pemilihan:**
>
> Multilayer Perceptron dipilih karena dataset Online Shoppers
> Purchasing Intention berbentuk **data tabular numerik** hasil encoding
> dan scaling. MLP mampu mempelajari hubungan non-linear antar fitur
> perilaku pengunjung website, sehingga cocok untuk tugas **klasifikasi
> biner** dalam memprediksi niat pembelian (purchase intention). Selain
> itu, MLP merupakan arsitektur deep learning yang paling umum dan
> efektif untuk data tabular dibandingkan CNN atau RNN yang lebih sesuai
> untuk citra dan data sekuensial.

2.  **Arsitektur Model**

> **Deskripsi Layer:**

1.  Input Layer : shape (17)

2.  Dense Layer 1 : Dense(128, activation=\'relu\')

3.  Dropout Layer 1 : Dropout(0.3)

4.  Dense Layer 2 : Dense(64, activation=\'relu\')

5.  Dropout Layer 2 : Dropout(0.3)

6.  Output Layer : Dense(1, activation=\'sigmoid\')

> Total parameters: 11.777
>
> Trainable parameters: 11.777

3.  **Input & Preprocessing Khusus**

> **Input shape:** (17,)
>
> **Preprocessing khusus untuk DL:**

-   Fitur boolean dan kategorikal tertentu dikonversi ke bentuk biner
    (0/1) agar seluruh input bersifat numerik.

-   Scaling menggunakan RobustScaler diterapkan pada fitur numerik
    tertentu yang memiliki outliers untuk menjaga stabilitas distribusi
    data.

-   Tidak dilakukan data augmentation karena dataset berupa data tabular
    dan tidak memerlukan teknik augmentasi khusus.

4.  **Hyperparameter**

> **Training Configuration:**

-   **Optimizer**: Adam

-   **Learning Rate**: 0.001

-   **Loss Function**: Binary Crossentropy

-   **Metrics**: Accuracy

-   **Batch Size**: 32

-   **Epochs**: 20

-   **Validation Split**: 0.2

-   **Callbacks**: EarlyStopping

5.  **Implementasi (Ringkas)**

> **Framework:** TensorFlow/Keras
>
> #@title Deep Learning Model : Multilayer Perceptron (MLP)
>
> import tensorflow as tf
>
> from tensorflow.keras.models import Sequential
>
> from tensorflow.keras.layers import Dense, Dropout, Input
>
> from tensorflow.keras.optimizers import Adam
>
> from tensorflow.keras.callbacks import EarlyStopping
>
> model_dl = Sequential(\[
>
>     Input(shape=(X_train.shape\[1\],)),
>
>     Dense(128, activation=\'relu\'),
>
>     Dropout(0.3),
>
>     Dense(64, activation=\'relu\'),
>
>     Dropout(0.3),
>
>     Dense(1, activation=\'sigmoid\')
>
> \])
>
> model_dl.compile(
>
>     optimizer=Adam(learning_rate=0.001),
>
>     loss=\'binary_crossentropy\',
>
>     metrics=\[\'accuracy\'\]
>
> )
>
> early_stopping = EarlyStopping(
>
>     monitor=\'val_loss\',
>
>     patience=5,
>
>     restore_best_weights=True
>
> )
>
> history = model_dl.fit(
>
>     X_train, y_train,
>
>     validation_data=(X_val, y_val),
>
>     epochs=20,
>
>     batch_size=32,
>
>     callbacks=\[early_stopping\],
>
>     verbose=1
>
> )

6.  **Training Process**

> **Training Time:**
>
> Proses training model deep learning berlangsung selama kurang lebih 40
> detik untuk 20 epoch.
>
> **Computational Resource:**
>
> Training dilakukan menggunakan **CPU** pada platform **Google Colab**.
>
> **Training History Visualization:**

1.  **Training & Validation Loss** per epoch

> ![](media/image4.png){width="4.593799212598425in"
> height="2.9640693350831144in"}

2.  **Training & Validation Accuracy/Metric** per epoch

> ![](media/image5.png){width="4.502027559055118in"
> height="2.9273162729658795in"}
>
> **Analisis Training:**

-   Apakah model mengalami overfitting?

> **Tidak.** Hal ini terlihat dari kurva *training loss* dan *validation
> loss* yang sama-sama menurun dan saling berdekatan hingga akhir epoch.
> Selain itu, nilai *training accuracy* dan *validation accuracy*
> menunjukkan pola yang konsisten tanpa perbedaan yang signifikan,
> sehingga tidak terdapat indikasi overfitting.

-   Apakah model sudah converge?

> **Ya.** Model menunjukkan penurunan loss yang signifikan pada beberapa
> epoch awal, kemudian kurva loss dan accuracy cenderung stabil pada
> epoch selanjutnya. Kondisi ini menandakan bahwa proses pembelajaran
> model telah mencapai titik konvergensi.

-   Apakah perlu lebih banyak epoch?

> **Tidak.** Penambahan jumlah epoch diperkirakan tidak akan memberikan
> peningkatan performa yang signifikan, karena nilai loss dan accuracy
> sudah stabil.

7.  **Model Summary**

> **Model: \"sequential_2\"**
>
> ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
>
> ┃ **Layer (type)** ┃ **Output Shape** ┃ **Param \#** ┃
>
> ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
>
> │ dense_6 (Dense) │ (None, 128) │ 3,456 │
>
> ├─────────────────────────────────┼────────────────────────┼───────────────┤
>
> │ dropout_4 (Dropout) │ (None, 128) │ 0 │
>
> ├─────────────────────────────────┼────────────────────────┼───────────────┤
>
> │ dense_7 (Dense) │ (None, 64) │ 8,256 │
>
> ├─────────────────────────────────┼────────────────────────┼───────────────┤
>
> │ dropout_5 (Dropout) │ (None, 64) │ 0 │
>
> ├─────────────────────────────────┼────────────────────────┼───────────────┤
>
> │ dense_8 (Dense) │ (None, 1) │ 65 │
>
> └─────────────────────────────────┴────────────────────────┴───────────────┘
>
> **Total params:** 35,333 (138.02 KB)
>
> **Trainable params:** 11,777 (46.00 KB)
>
> **Non-trainable params:** 0 (0.00 B)
>
> **Optimizer params:** 23,556 (92.02 KB)

7.  **EVALUATION**

    1.  **Metrik Evaluasi**

-   **Accuracy**: untuk mengukur ketepatan prediksi secara keseluruhan.

-   **Precision**: untuk menilai ketepatan prediksi kelas positif.

-   **Recall**: untuk mengukur kemampuan model mendeteksi seluruh data
    > kelas positif.

-   **F1-Score**: sebagai ukuran keseimbangan antara precision dan
    > recall.

-   **Confusion Matrix**: untuk menganalisis kesalahan klasifikasi
    > secara lebih detail.

    1.  **Hasil Evaluasi Model**

1.  **Model 1 (Baseline)**

> **Metrik:**

-   **Accuracy:** 0.891

-   **Precision:** 0.79

-   **Recall:** 0.42

-   **F1-Score:** 0.55

> **Confusion Matrix / Visualization:**
>
> ![](media/image6.png){width="4.134718941382327in"
> height="3.3111679790026245in"}

2.  **Model 2 (Advanced/ML)**

> **Metrik:**

-   **Accuracy:** 0.908

-   **Precision:** 0.77

-   **Recall:** 0.60

-   **F1-Score:** 0.67

> **Confusion Matrix / Visualization:**
>
> ![](media/image7.png){width="4.0258716097987755in"
> height="3.6311811023622047in"}
>
> **Feature Importance (jika applicable):**
>
> ![](media/image8.png){width="5.149024496937883in"
> height="2.802280183727034in"}

3.  **Model 3 (Deep Learning)**

> **Metrik:**

-   **Accuracy:** 0.904

-   **Precision:** 0.78

-   **Recall:** 0.54

-   **F1-Score:** 0.64

> **Confusion Matrix / Visualization:**
>
> ![](media/image9.png){width="4.105589457567804in"
> height="3.5912423447069117in"}
>
> **Training History:**
>
> ![](media/image10.png){width="5.740387139107612in"
> height="4.1697353455818025in"}
>
> **Test Set Predictions:**
>
> ![](media/image11.png){width="2.4734569116360454in"
> height="3.339582239720035in"}

1.  **Perbandingan Ketiga Model**

> **Tabel Perbandingan:**

  -------------------------------------------------------------------------------------------------------
  **Model**         **Accuracy**   **Precision**   **Recall**   **F1-Score**   **Training   **Inference
                                                                               Time**       Time**
  ----------------- -------------- --------------- ------------ -------------- ------------ -------------
  Baseline          0.892          0.84            0.70         0.74           3 s          \~0.01 s
  (Logistic                                                                                 
  Regression)                                                                               

  Advanced (Random  0.908          0.85            0.78         0.81           3 s          \~0.01 s
  Forest)                                                                                   

  Deep Learning     0.905          0.85            0.76         0.79           40 s         \~0.10 s
  (MLP)                                                                                     
  -------------------------------------------------------------------------------------------------------

> **Visualisasi Perbandingan:**
>
> ![](media/image12.png){width="5.752931977252843in"
> height="3.412215660542432in"}

2.  **Analisis Hasil**

> **Interpretasi:**

1.  **Model Terbaik:**

> Model **Random Forest** dipilih sebagai model terbaik karena memiliki
> nilai accuracy (0.908), precision, recall, dan F1-score tertinggi
> dibandingkan model lain, sehingga menunjukkan keseimbangan performa
> yang paling baik.

2.  **Perbandingan dengan Baseline:**

> Dibandingkan Logistic Regression sebagai baseline, Random Forest dan
> MLP menunjukkan peningkatan performa yang jelas, terutama pada recall
> dan F1-score, yang menandakan kemampuan lebih baik dalam mendeteksi
> kelas positif.

3.  **Trade-off:**

-   **Logistic Regresion:**

    -   Paling sederhana, cepat dilatih, dan inference sangat cepat.

    -   Performa paling rendah.

-   **Random Forest:**

    -   Performa terbaik dengan waktu training masih relatif cepat.

    -   Model lebih kompleks namun masih interpretatif.

-   **Deep Learning:**

    -   Performa mendekati Random Forest.

    -   Membutuhkan waktu training paling lama dan komputasi lebih
        tinggi.

4.  **Error Analysis:**

-   **False Positive**: Model memprediksi pengunjung akan membeli,
    tetapi tidak terjadi pembelian. Biasanya muncul pada pengguna dengan
    durasi kunjungan lama atau banyak melihat produk tanpa melakukan
    transaksi.

-   **False Negative**: Model memprediksi tidak membeli, padahal terjadi
    pembelian. Kesalahan ini sering terjadi pada pembelian cepat atau
    impulsif dengan interaksi singkat.

-   **Implikasi**: False negative lebih krusial karena berpotensi
    menyebabkan kehilangan peluang konversi, sehingga peningkatan recall
    kelas positif masih diperlukan.

5.  **Overfitting/Underfitting:**

> Berdasarkan hasil evaluasi dan perbandingan performa, ketiga model
> **tidak menunjukkan indikasi overfitting atau underfitting** yang
> signifikan. Performa pada data uji relatif konsisten, menandakan
> kemampuan generalisasi model yang cukup baik.

8.  **CONCLUSION**

    1.  **Kesimpulan Utama**

> **Model Terbaik:** Random Forest
>
> **Alasan:**

-   Random Forest menghasilkan akurasi tertinggi dibandingkan Logistic
    Regression dan MLP.

-   Model ini memiliki keseimbangan precision, recall, dan F1-score yang
    lebih baik, khususnya pada kelas positif.

-   Mampu menangani hubungan non-linear antar fitur tanpa memerlukan
    preprocessing yang kompleks.

-   Lebih stabil terhadap outliers dan variasi data, sehingga
    performanya konsisten.

-   Waktu training dan inference masih efisien dibandingkan model deep
    learning yang lebih kompleks.

> **Pencapaian Goals:**

  ----------------------------------------------------------------------------------
  **No**   **Goals**                    **Tercapai**   **Penjelasan Singkat**
  -------- ---------------------------- -------------- -----------------------------
  1        Membangun model machine                     Seluruh model (Logistic
           learning dengan akurasi                     Regression, Random Forest,
           minimal 80%                                 dan MLP) mencapai akurasi di
                                                       atas 80%.

  2        Membandingkan performa tiga                 Evaluasi dilakukan
           pendekatan model menggunakan                menggunakan accuracy,
           metrik evaluasi                             precision, recall, dan
                                                       F1-score pada ketiga model.

  3        Menentukan model terbaik                    Random Forest menunjukkan
           dalam mengenali pola                        performa terbaik secara
           perilaku pengguna                           konsisten berdasarkan metrik
                                                       evaluasi.

  4        Menghasilkan proses                         Proses preprocessing,
           pengolahan data dan                         splitting data, dan training
           pelatihan yang reproducible                 dilakukan dengan random state
                                                       yang tetap.
  ----------------------------------------------------------------------------------

1.  **Key Insights**

> **Insight dari Data:**

-   Perilaku pengguna seperti durasi kunjungan, jumlah halaman yang
    > dilihat, dan nilai interaksi (page value) memiliki hubungan kuat
    > dengan keputusan pembelian.

-   Dataset memiliki ketidakseimbangan kelas, di mana pengunjung yang
    > tidak melakukan pembelian lebih dominan dibandingkan pembeli.

-   Outliers pada fitur durasi dan interaksi merupakan perilaku alami
    > pengguna dan tetap menyimpan informasi penting.

> **Insight dari Modeling:**

-   Model ensemble (Random Forest) memberikan performa paling stabil dan
    > konsisten dibandingkan baseline dan deep learning.

-   Deep Learning mampu menangkap pola non-linear, namun membutuhkan
    > waktu training lebih lama dengan peningkatan performa yang tidak
    > signifikan dibanding Random Forest.

-   Baseline model (Logistic Regression) cukup baik sebagai pembanding
    > awal, tetapi kurang optimal dalam menangkap pola kompleks pada
    > data.

    1.  **Kontribusi Proyek**

> **Manfaat praktis:**

-   Model dapat digunakan untuk memprediksi peluang pembelian pengunjung
    website, sehingga membantu strategi pemasaran digital seperti
    personalisasi promosi dan retargeting.

-   Hasil prediksi dapat dimanfaatkan untuk mengoptimalkan konversi
    penjualan berbasis data perilaku pengguna.

> **Pembelajaran yang didapat:**

-   Memahami pentingnya data preparation dalam meningkatkan performa
    model.

-   Mempelajari perbandingan performa antara baseline, ensemble, dan
    deep learning pada data tabular.

-   Memahami trade-off antara akurasi, kompleksitas model, dan waktu
    komputasi dalam machine learning.

9.  **FUTURE WORK (Opsional)**

> Saran pengembangan untuk proyek selanjutnya: \*\* Centang Sesuai
> dengan saran anda \*\*
>
> **Data:**

-   Mengumpulkan lebih banyak data

-   Menambah variasi data

-   Feature engineering lebih lanjut

> **Model:**

-   Mencoba arsitektur DL yang lebih kompleks

-   Hyperparameter tuning lebih ekstensif

```{=html}
<!-- -->
```
-   Ensemble methods (combining models)

-   Transfer learning dengan model yang lebih besar

> **Deployment:**

-   Membuat API (Flask/FastAPI)

-   Membuat web application (Streamlit/Gradio)

```{=html}
<!-- -->
```
-   Containerization dengan Docker

-   Deploy ke cloud (Heroku, GCP, AWS)

> **Optimization:**

-   Model compression (pruning, quantization)

```{=html}
<!-- -->
```
-   Improving inference speed

```{=html}
<!-- -->
```
-   Reducing model size

10. **REPRODUCIBILITY (WAJIB)**

    1.  **GitHub Repository**

> **Link
> Repository:** <https://github.com/balqissph/234311008_Balqis_UAS_Data-Science>
>
> **Repository harus berisi:**

-   ✅ Notebook Jupyter/Colab dengan hasil running

-   ✅ Script Python (jika ada)

-   ✅ requirements.txt atau environment.yml

-   ✅ README.md yang informatif

-   ✅ Folder structure yang terorganisir

-   ✅ .gitignore (jangan upload dataset besar)

    1.  **Environment & Dependencies**

> **Python Version:** 3.12.12
>
> **Main Libraries & Versions:**
>
> numpy==2.02
>
> pandas==2.2.2
>
> scikit-learn==1.6.1
>
> matplotlib==3.10.0
>
> seaborn==0.13.2
>
> Deep Learning Framework:
>
> tensorflow==2.19.0
>
> keras==3.10.0
