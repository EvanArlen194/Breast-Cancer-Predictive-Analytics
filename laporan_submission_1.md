# Laporan Proyek Machine Learning - Evan Arlen Handy

## Domain Proyek

**Domain proyek yang dipilih dalam proyek ini adalah mengenai tema kesehatan dengan judul proyek "Klasifikasi Kanker Payudara Menggunakan Algoritma Machine Learning untuk Diagnosis Awal"**.

### Latar Belakang
Kanker adalah salah satu penyebab utama kematian di seluruh dunia. Berdasarkan data GLOBOCAN (Global Burden of Cancer), International Agency for Research on Cancer (IARC), kita mengetahui bahwa pada tahun 2020 terdapat 19.292.789 kasus kanker baru dan 9.958.133 kematian akibat kanker di seluruh dunia. Kasus kanker tahunan diperkirakan meningkat dari 14 juta menjadi 22 juta selama dua dekade mendatang. WHO memperkirakan pada tahun 2040, angka kejadian kanker akan mencapai 28 juta orang. [[1](https://ojs.uho.ac.id/index.php/JIMKESMAS/article/view/2879)]

Kanker payudara menempati urutan pertama dalam jumlah kasus kanker dan merupakan penyebab utama kematian akibat kanker di seluruh dunia setiap tahunnya. Menurut WHO (2020), angka kejadian kanker payudara sebanyak 2.261.419 kasus, dimana kanker ini terutama menyerang wanita. Angka kejadian di negara berkembang 88% lebih tinggi dibandingkan di negara maju (masing-masing 55,9 dan 29,7 per 100.000) dan angka kematian sebesar 17%. Angka kejadian penyakit ini diperkirakan semakin meningkat di seluruh dunia. Sementara menurut data GLOBOCAN tahun 2020, diketahui bahwa kanker payudara merupakan jenis kanker dengan angka kejadian tertinggi sebesar 11% dan angka kematian akibat kanker payudara sebesar 6,9%. Di Indonesia, kanker payudara merupakan jenis kanker dengan angka kejadian tertinggi kedua setelah kanker serviks dan cenderung meningkat setiap tahunnya. Kebanyakan tumor payudara ganas terjadi pada stadium lanjut. Jumlah kasus kanker payudara di Indonesia kurang lebih 65.858 kasus baru per tahun (populasi 273.523.621). Berdasarkan data Riskesdas tahun 2018, di Indonesia angka kejadian penyakit kanker sebanyak 1.017.290 kasus dan di wilayah Sulawesi Selatan sebanyak 33.693 kasus. [[1](https://ojs.uho.ac.id/index.php/JIMKESMAS/article/view/2879)] [[2](https://repository.badankebijakan.kemkes.go.id/id/eprint/3514/)] [[3](https://jurnal.fk.umi.ac.id/index.php/umimedicaljournal/article/view/34)]

Faktor risiko yang sangat terkait dengan peningkatan kejadian kanker payudara meliputi jenis kelamin wanita, usia > 50 tahun, riwayat keluarga dan genetika (pembawa BRCA1, BRCA2, ATM atau TP53 (p53)), riwayat penyakit payudara (DCIS pada payudara yang sama, LCIS, kepadatan tinggi pada mamografi), riwayat menarche/menarche dini (<12 tahun) atau menopause terlambat (>55 tahun), riwayat reproduksi (menyusui dan tidak menyusui), hormon, obesitas, konsumsi alkohol, riwayat radiasi dinding dada, dan faktor lingkungan. Epidemiologi sedang bergeser dari penyakit menular ke penyakit tidak menular dan kejadian kanker payudara meningkat secara global. [[4](http://jurnal.fk.unand.ac.id/index.php/jka/article/view/965)]

Berdasarkan latar belakang diatas kanker payudara merupakan penyebab utama kematian pada wanita di seluruh dunia. Deteksi dini kanker payudara sangat penting karena memungkinkan pengobatan yang lebih efektif dan dapat meningkatkan peluang bertahan hidup. Sayangnya, metode konvensional seperti biopsi membutuhkan waktu dan dapat menyebabkan kecemasan bagi pasien. Oleh karena itu, pengembangan sistem berbasis machine learning yang dapat membantu dokter dalam mendiagnosis kanker payudara menjadi sangat penting. Dengan adanya proyek ini diharapkan dapat membantu mengklasifikasikan tumor payudara sebagai jinak atau ganas berdasarkan data klinis dari dataset **Breast Cancer Wisconsin (Diagnostic) Data Set** [[5]](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) yang berisi karakteristik tumor dan label diagnosis, sehingga dapat mempercepat proses diagnosis dan meningkatkan akurasi.

Referensi: 
---
- [[1]](https://ojs.uho.ac.id/index.php/JIMKESMAS/article/view/2879) Novia Wulandari, Hartati Bahar, dan Cece Suriani Ismail. 2017. Gambaran Kualitas Hidup Pada Penderita Kanker Payudara di Rumah Sakit Umum Bahteramas Provinsi Sulawesi Tenggara. JIMKESMAS, 2(6).
- [[2]](https://repository.badankebijakan.kemkes.go.id/id/eprint/3514/) Tim Riskerdas 2018. 2019. Laporan Nasional Riskesdas 2018. Jakarta: Lembaga Penelitian Badan Penelitian dan Pengembangan.
- [[3]](https://jurnal.fk.umi.ac.id/index.php/umimedicaljournal/article/view/34) Faisal Sommeng. 2019. Hubungan Status Fisik Pra Anastesi Umum Dengan Waktu Pulih Sadar Pasien Pasca Operasi Mastektomi Di RS Ibnu Sina Februari - Maret 2017. UMI Medical Journal, 3(1). Hal: 47-58.
- [[4]](http://jurnal.fk.unand.ac.id/index.php/jka/article/view/965) Rusydah Syarlina, Azamris, Avit Suchitra, dan Wirsma Arif Harahap. 2019. Hubungan Interval Waktu Antara Usia Menarche dan Usia Saat Melahirkan Anak Pertama Cukup Bulan dengan Kejadian Kanker Payudara di RSUP dr. M. Djamil Padang Pada Tahun 2014-2017. Jurnal Kesehatan Andalas, 8(1).
- [[5]](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) Breast Cancer Data Set - Kaggle



## Business Understanding
Dalam proyek ini, peneliti menganalisis dan mengklasifikasikan tumor kanker payudara menjadi dua kategori: Malignant (Ganas) dan Benign (Jinak). Dengan menggunakan dataset **[Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)**, peneliti menerapkan beberapa algoritma machine learning untuk mengidentifikasi dan mengklasifikasikan tumor berdasarkan fitur-fitur medis yang tersedia.
### Problem Statements
Berdasarkan latar belakang yang sudah dipaparkan di atas, rumusan masalah yang diperoleh sebagai berikut:
- Bagaimana cara mengklasifikasikan jenis tumor (Malignant atau Benign) secara akurat menggunakan fitur-fitur medis pada dataset yang ada?

- Bagaimana cara menangani masalah ketidakseimbangan kelas (imbalanced class) antara tumor Malignant dan Benign, dan memastikan model tetap memberikan performa yang baik pada kedua kategori?

- Model machine learning mana yang memberikan akurasi terbaik dalam mengklasifikasikan kanker payudara, dan bagaimana cara meningkatkan performa model tersebut?

### Goals
- Tujuan utama adalah mengembangkan model machine learning yang mampu mengklasifikasikan tumor sebagai Malignant atau Benign dengan akurasi tinggi berdasarkan fitur-fitur medis yang tersedia dalam dataset.
- Untuk menangani ketidakseimbangan kelas adalah dengan teknik oversampling seperti Synthetic Minority Over-sampling Technique (SMOTE) agar distribusi kelas lebih seimbang, sehingga model tidak bias terhadap kelas mayoritas.
- Peneliti akan membandingkan beberapa model machine learning (Support Vector Classifier, K-Nearest Neighbors, dan Decision Tree) berdasarkan akurasi dan laporan klasifikasi mereka. Model terbaik akan dipilih berdasarkan metrik akurasi tertinggi.

### Solution Statements
Untuk mencapai tujuan ini, peneliti mengusulkan beberapa solusi yang akan diuji dan dibandingkan:
- Menggunakan tiga algoritma machine learning utama (Support Vector Classifier, K-Nearest Neighbors, dan Decision Tree) untuk melatih model. Setiap algoritma akan dioptimalkan dan dievaluasi menggunakan akurasi dan laporan klasifikasi.
- Melakukan preprocessing data dengan menangani pencilan menggunakan metode EllipticEnvelope dan menormalkan data menggunakan StandardScaler untuk memastikan model menerima data yang optimal.
- Menerapkan SMOTE untuk mengatasi masalah ketidakseimbangan kelas, dengan tujuan meningkatkan performa klasifikasi pada kelas minoritas (Malignant).
- Mengukur kinerja setiap model dengan akurasi serta laporan klasifikasi, termasuk metrik precision, recall, dan F1-score. Model dengan akurasi tertinggi dan metrik evaluasi terbaik akan dipilih sebagai model final.

## Data Understanding

### Informasi Dataset

Dataset yang digunakan dalam proyek ini adalah **[Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)**. Dataset ini berisi informasi medis mengenai berbagai sampel tumor payudara yang diklasifikasikan sebagai **Malignant (M)** atau **Benign (B)**. Dataset ini banyak digunakan dalam penelitian machine learning untuk tugas klasifikasi kanker payudara.

### Deskripsi Variabel

Dataset ini terdiri dari 569 baris dan 32 kolom, di mana setiap baris merepresentasikan satu sampel tumor payudara. Berikut adalah deskripsi dari masing-masing variabel yang terdapat dalam dataset:

| **Variabel**                | **Deskripsi**                                                                                                                                               |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `id`                        | Identifier unik untuk setiap sampel. Kolom ini tidak relevan untuk analisis dan akan dihapus selama pra-pemrosesan data.                                      |
| `diagnosis`                 | Label target yang menunjukkan apakah tumor bersifat **Malignant (M)** atau **Benign (B)**. Ini adalah variabel yang akan diprediksi oleh model.               |
| `radius_mean`               | Rata-rata jarak dari pusat ke titik pada keliling tumor.                                                                                                     |
| `texture_mean`              | Rata-rata standar deviasi nilai skala abu-abu (texture) pada permukaan tumor.                                                                               |
| `perimeter_mean`            | Rata-rata keliling tumor.                                                                                                                                   |
| `area_mean`                 | Rata-rata luas tumor.                                                                                                                                         |
| `smoothness_mean`           | Rata-rata tingkat kehalusan permukaan tumor.                                                                                                                |
| `compactness_mean`          | Rata-rata kekompakan permukaan tumor, dihitung sebagai keliling² dibagi luas minus 1.                                                                       |
| `concavity_mean`            | Rata-rata tingkat kecekungan permukaan tumor.                                                                                                               |
| `concave points_mean`       | Rata-rata jumlah titik cekung pada permukaan tumor.                                                                                                        |
| `symmetry_mean`             | Rata-rata tingkat simetri tumor.                                                                                                                              |
| `fractal_dimension_mean`    | Rata-rata dimensi fraktal dari permukaan tumor, yang mengukur kompleksitas permukaan.                                                                        |
| `radius_se`                 | Standard error dari `radius_mean`.                                                                                                                             |
| `texture_se`                | Standard error dari `texture_mean`.                                                                                                                            |
| `perimeter_se`              | Standard error dari `perimeter_mean`.                                                                                                                          |
| `area_se`                   | Standard error dari `area_mean`.                                                                                                                               |
| `smoothness_se`             | Standard error dari `smoothness_mean`.                                                                                                                         |
| `compactness_se`            | Standard error dari `compactness_mean`.                                                                                                                        |
| `concavity_se`              | Standard error dari `concavity_mean`.                                                                                                                          |
| `concave points_se`         | Standard error dari `concave points_mean`.                                                                                                                   |
| `symmetry_se`               | Standard error dari `symmetry_mean`.                                                                                                                            |
| `fractal_dimension_se`      | Standard error dari `fractal_dimension_mean`.                                                                                                                 |
| `radius_worst`              | Nilai terburuk (maksimal) dari `radius_mean`.                                                                                                                 |
| `texture_worst`             | Nilai terburuk dari `texture_mean`.                                                                                                                            |
| `perimeter_worst`           | Nilai terburuk dari `perimeter_mean`.                                                                                                                         |
| `area_worst`                | Nilai terburuk dari `area_mean`.                                                                                                                              |
| `smoothness_worst`          | Nilai terburuk dari `smoothness_mean`.                                                                                                                        |
| `compactness_worst`         | Nilai terburuk dari `compactness_mean`.                                                                                                                       |
| `concavity_worst`           | Nilai terburuk dari `concavity_mean`.                                                                                                                         |
| `concave points_worst`      | Nilai terburuk dari `concave points_mean`.                                                                                                                    |
| `symmetry_worst`            | Nilai terburuk dari `symmetry_mean`.                                                                                                                          |
| `fractal_dimension_worst`   | Nilai terburuk dari `fractal_dimension_mean`.                                                                                                                 |

### Exploratory Data Analysis (EDA)

Untuk memahami lebih dalam mengenai dataset, beberapa tahapan eksplorasi data dilakukan sebagai berikut:

1. **Pemeriksaan Dimensi dan Tipe Data**:
   - Menggunakan `data_breast.info()` untuk melihat jumlah baris dan kolom, serta tipe data dari setiap variabel.

     <img width="217" alt="data_breast info()" src="https://github.com/user-attachments/assets/b56240e7-d877-49f0-a970-ca3b220e6810">

     Fungsi ini digunakan untuk mendapatkan informasi ringkas tentang DataFrame data_breast. Hasil keluaran dari fungsi ini 
     mencakup:
      - baris dan kolom: Memberikan gambaran umum tentang ukuran dataset.
      - Tipe data dari setiap variabel: Menunjukkan tipe data dari setiap kolom, seperti integer, float, dan objek. Ini 
        penting untuk memahami jenis analisis atau transformasi yang mungkin perlu dilakukan pada data.
      - Jumlah nilai non-null: Menunjukkan berapa banyak entri yang tidak kosong dalam setiap kolom, yang membantu dalam            mendeteksi missing values (nilai yang hilang).
   - Menggunakan `data_breast.describe()`
     Fungsi ini memberikan statistik deskriptif dari DataFrame, khususnya untuk kolom-kolom numerik. Hasil yang ditampilkan      mencakup:

      - Mean (rata-rata): Nilai rata-rata dari setiap kolom.
      - Median (nilai tengah): Nilai yang membagi dataset menjadi dua bagian yang sama.
      - Standar deviasi: Mengukur seberapa tersebar nilai-nilai dari rata-rata. Semakin besar nilai standar deviasi,       
        semakin besar variasi data.
      - Minimum dan maksimum: Nilai terendah dan tertinggi dari setiap kolom. Ini membantu untuk memahami rentang nilai 
        dari fitur.
      - Kuartil: Nilai yang membagi data menjadi empat bagian yang sama, membantu untuk memahami distribusi data.
   - Menggunakan `data_breast.head()`
     Fungsi ini menampilkan beberapa baris pertama dari DataFrame data_breast (secara default 5 baris). Kegunaannya adalah:

     - Melihat data secara langsung: Memungkinkan pengguna untuk mendapatkan gambaran langsung tentang nilai-nilai dalam    
       dataset.
     - Memeriksa format dan konten: Membantu dalam memverifikasi bahwa data telah dimuat dengan benar dan sesuai dengan 
       harapan.
     - Memastikan bahwa kolom dan data yang relevan: Dapat digunakan untuk memastikan bahwa kolom yang diharapkan ada dan 
       memiliki data yang benar.
   - Menggunakan `data_breast.isnull().sum()` untuk memeriksa apakah terdapat nilai yang hilang dalam dataset. Jika ada missing values, output akan menunjukkan jumlah nilai yang hilang untuk setiap kolom. Jika ada missing values, perlu dilakukan penanganan seperti menghapus, mengisi dengan nilai rata-rata, atau menggunakan teknik imputasi.
     
     <img width="134" alt="missing-value" src="https://github.com/user-attachments/assets/34850651-c356-41d2-a374-637e86bc371d">

   - Menggunakan `data_breast.duplicated().sum()` untuk menghitung jumlah baris yang merupakan duplikat dalam dataset. Data duplikat dapat menyebabkan bias dalam model dan analisis, sehingga penting untuk memeriksanya. Jika ditemukan, langkah selanjutnya biasanya adalah menghapus baris-baris duplikat untuk memastikan keakuratan analisis.
     
      <img width="90" alt="duplikat" src="https://github.com/user-attachments/assets/a4f82ab9-fa5a-4aad-88a2-e6373316a62e">

   - Bagian ini fokus pada deteksi outlier menggunakan metode Interquartile Range (IQR):
     - Pemilihan Kolom Numerik: Hanya kolom yang memiliki tipe data numerik (float64 dan int64) yang dipilih untuk analisis outlier. Hal ini penting karena IQR hanya relevan untuk data numerik.
     - Perhitungan IQR:
     Q1 (kuartil pertama) dan Q3 (kuartil ketiga) dihitung untuk menentukan rentang interkuartil.
     IQR dihitung sebagai selisih antara Q3 dan Q1.
     - Batas Bawah dan Atas: 
     Menghitung batas bawah dan atas untuk menentukan nilai-nilai yang dianggap sebagai outlier. Nilai di bawah batas bawah atau di atas batas atas dianggap sebagai outlier.
     - Cek Jumlah Outlier: 
     Jumlah outlier dihitung untuk setiap fitur dengan membandingkan nilai-nilai dalam kolom numerik terhadap batas bawah dan atas. Hasilnya ditampilkan untuk memberikan gambaran tentang jumlah outlier yang ada di setiap fitur.

      <img width="143" alt="outlier" src="https://github.com/user-attachments/assets/1c21f8d0-d12d-4243-8fae-8f0ab23cb24b">

1. **Visualisasi Distribusi Kelas Target**:
   - Menggunakan `sns.countplot()` untuk memvisualisasikan distribusi antara kelas **Malignant** dan **Benign**. Hal ini penting untuk memahami keseimbangan kelas dalam dataset. Dengan menggunakan countplot, kita dapat melihat proporsi antara kategori Malignant (kanker ganas) dan Benign (kanker jinak) dalam data. Visualisasi ini menunjukkan ketidakseimbangan antara kedua kelas, yang penting untuk dipertimbangkan saat memilih teknik pemodelan dan evaluasi. Misalnya, jika salah satu kelas jauh lebih banyak daripada yang lain, ini dapat mempengaruhi hasil model, sehingga teknik seperti SMOTE mungkin diperlukan untuk menyeimbangkan data.

![countplot](https://github.com/user-attachments/assets/47a456a0-f76f-4233-94f7-bac9f1714487)

3. **Visualisasi Distribusi Fitur Numerik**:
   - Menggunakan boxplot untuk melihat distribusi dari setiap fitur numerik. Ini membantu dalam mengidentifikasi outliers dan memahami sebaran data. boxplot digunakan untuk memeriksa distribusi fitur numerik setelah penghapusan pencilan. Boxplot memberikan gambaran jelas mengenai rentang, median, dan variabilitas dari fitur-fitur tersebut. Dengan menganalisis boxplot, kita dapat mengidentifikasi apakah ada fitur yang memiliki distribusi yang sangat berbeda, serta melihat apakah ada nilai ekstrem yang mungkin perlu ditangani.

![boxplot_pencilan](https://github.com/user-attachments/assets/a96bb8d5-b5fb-4e73-ba47-4f807450dff5)


4. **Analisis Korelasi Antar Fitur**:
   - Menggunakan heatmap korelasi (`sns.heatmap()`) untuk melihat hubungan antar fitur. Fitur-fitur yang memiliki korelasi tinggi dapat mempengaruhi pemilihan fitur atau teknik reduksi dimensi.

![sns_heatmap](https://github.com/user-attachments/assets/45ad2012-b88a-44e0-9c30-3a928d5fbc6e)

### Kesimpulan

Melalui tahapan Data Understanding dan Exploratory Data Analysis di atas, peneliti berhasil memahami struktur dan karakteristik dataset kanker payudara. Proses ini membantu dalam mengidentifikasi langkah-langkah pra-pemrosesan yang diperlukan serta memberikan wawasan mengenai fitur-fitur yang paling berpengaruh dalam klasifikasi tumor sebagai Malignant atau Benign. Pemahaman mendalam terhadap data ini merupakan dasar yang kuat untuk pengembangan model machine learning yang akurat dan handal.

## Data Preparation

Data preparation adalah langkah penting sebelum membangun model machine learning. Tujuan dari tahapan ini adalah memastikan bahwa data siap untuk digunakan dalam proses pelatihan model dengan cara melakukan pembersihan data, transformasi, dan pemilihan fitur yang relevan. Berikut adalah teknik-teknik data preparation yang telah dilakukan:

### 1. Menghapus Kolom yang Tidak Relevan
Kolom `id` dan `Unnamed: 32` dihapus karena tidak memberikan informasi penting bagi model dalam menentukan diagnosis. Kolom `id` hanya digunakan sebagai pengidentifikasi unik untuk setiap sampel, serta kolom `Unnamed: 32` tidak memiliki hubungan langsung dengan diagnosis tumor dan hanya akan menambah noise pada model, sehingga perlu dihapus. Sehingga tidak memiliki pengaruh terhadap prediksi.

**Langkah**:
```python
data_breast = data_breast.drop(columns=['Unnamed: 32', 'id'], axis=1)
```

**Dataset Sebelum Menghapus Kolom**

<img width="235" alt="sebelum_hapus_kolom" src="https://github.com/user-attachments/assets/6300d60c-282b-4569-b6fe-ec9d01300c38">

**Dataset Sesudah Menghapus Kolom**

<img width="243" alt="setelah_hapus_kolom" src="https://github.com/user-attachments/assets/912657a9-35f3-4523-881a-de63b227eca7">

### 2. Mendeteksi dan Menangani Pencilan
Pencilan akan dideteksi dan dihapus menggunakan **EllipticEnvelope**.
- **EllipticEnvelope** digunakan untuk mendeteksi pencilan berdasarkan asumsi distribusi Gaussian multivariat.
![multivariate-gaussian-models](https://github.com/user-attachments/assets/a027470a-42c7-4af4-a833-0c4a4158d955)
- Data dengan pencilan akan dihapus agar tidak mempengaruhi model yang akan dilatih.

**Langkah**:
```python
numerical_features = data_breast.select_dtypes(include=['float64', 'int64']).columns

# Menggunakan EllipticEnvelope untuk mendeteksi dan menangani pencilan
outlier_detector = EllipticEnvelope(contamination=0.01)

# Menghapus pencilan berdasarkan fitur numerik
for feature in numerical_features:
    feature_data = data_breast[[feature]].values
    mask = outlier_detector.fit_predict(feature_data)
    data_breast = data_breast[mask == 1]

print(f"Ukuran dataset setelah pencilan dihapus: {data_breast.shape}")
```
<img width="223" alt="ukuran-dataset-setelah-pencilan-dihapus" src="https://github.com/user-attachments/assets/03fb0c36-0b94-4e99-8f5b-db92a0b3292c">

### 3. Memetakan Kelas Target
Kolom `diagnosis` berisi label kategoris dalam bentuk huruf, yaitu M untuk malignant (ganas) dan B untuk benign (jinak). Algoritma machine learning umumnya memerlukan data dalam format numerik. Untuk memudahkan pemodelan, label ini diubah menjadi nilai numerik, di mana M menjadi 1 dan B menjadi 0.

**Langkah**:
```python
data_breast['diagnosis'] = data_breast['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
```

### 4. Memisahkan Variabel Fitur dan Target
Memisahkan antara variabel fitur (X) dan variabel target (y). Semua kolom selain diagnosis dianggap sebagai fitur (X), sedangkan diagnosis adalah target (y). Langkah ini dilakukan untuk mempersiapkan data fitur yang akan digunakan untuk melatih model dan target yang akan diprediksi oleh model.

**Langkah**
```python
X = data_breast.drop(['diagnosis'], axis=1)
y = data_breast['diagnosis']
```

### 5. Membagi Data ke dalam Training dan Test Set
Dataset dibagi menjadi dua bagian: Training Set untuk melatih model dan Test Set untuk mengevaluasi performa model. Peneliti menggunakan 70% data untuk pelatihan dan 30% untuk pengujian. Pembagian dataset penting untuk menghindari overfitting. Dengan membagi dataset, kita bisa mengevaluasi performa model pada data yang tidak dilihat saat pelatihan. Setelah membagi dataset peneliti memeriksa distribusi kelas di data pelatihan (y_train) untuk melihat apakah memiliki proporsi yang seimbang antara kelas yang ada.

**Langkah**:
``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Distribusi kelas di y_train:")
print(y_train.value_counts())
```
<img width="166" alt="y_train" src="https://github.com/user-attachments/assets/11d39934-9b9e-4beb-b4fe-b16ea2178111">

### 6. Menggunakan SMOTE untuk Oversampling
Sebelum menerapkan SMOTE (Synthetic Minority Over-sampling Technique), peneliti memeriksa apakah ada lebih dari satu kelas dalam y_train. Jika ada, peneliti menggunakan SMOTE untuk menyeimbangkan kelas dengan membuat contoh sintetis dari kelas minoritas. Jika SMOTE diterapkan, peneliti akan melihat distribusi kelas di y_train_balanced untuk memastikan bahwa kelas sudah seimbang setelah proses oversampling.
```python
# Memastikan ada lebih dari satu kelas
if len(np.unique(y_train)) > 1:
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Melihat Distribusi Kelas Setelah Oversampling
    print("Distribusi kelas setelah SMOTE:")
    print(pd.Series(y_train_balanced).value_counts())
else:
    print("Hanya ada satu kelas di y_train, tidak ada SMOTE yang diterapkan.")
```
<img width="182" alt="kelas_setelah_SMOTE" src="https://github.com/user-attachments/assets/108e137f-cd7a-4fa1-8df3-54f96e766858">

### 7. Normalisasi Data
Karena beberapa algoritma machine learning seperti Support Vector Machine (SVM) dan K-Nearest Neighbors (KNN) sensitif terhadap skala data, peneliti melakukan normalisasi menggunakan StandardScaler. Teknik ini mengubah fitur sehingga memiliki distribusi dengan rata-rata 0 dan standar deviasi 1. Hal ini karena normalisasi membantu memastikan bahwa fitur dengan skala yang lebih besar tidak mendominasi fitur dengan skala yang lebih kecil. Ini sangat penting untuk algoritma yang menghitung jarak antar data seperti KNN dan SVM.

![standard_scaler](https://github.com/user-attachments/assets/4d6d3156-ce5a-463b-b9b9-fe730de6eb7f)

**Langkah**:
``` python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Modeling

Pada proyek ini, beberapa algoritma machine learning diterapkan untuk menyelesaikan masalah klasifikasi kanker payudara, yaitu memprediksi apakah tumor bersifat **Malignant (ganas)** atau **Benign (jinak)**. Berikut adalah model-model yang digunakan beserta tahapan dan parameter yang diaplikasikan:

### 1. Support Vector Classifier (SVC)
- **Tahapan**: 
  SVC bekerja dengan mencari hyperplane terbaik yang memisahkan kelas. Model ini menggunakan kernel linier untuk membuat keputusan.
  - **Parameter utama**: C (regularization parameter) dan kernel.
  - **Scaling data**: SVC sensitif terhadap skala fitur, sehingga data dinormalisasi menggunakan **StandardScaler**.
- **Kelebihan**: 
  - Cocok untuk klasifikasi dengan batasan yang jelas antara kelas-kelas.
  - Bekerja dengan baik pada dataset berdimensi tinggi.
- **Kekurangan**: 
  - Tidak efisien pada dataset yang sangat besar.
  - Membutuhkan waktu komputasi yang lebih lama dibandingkan model lain seperti Decision Tree.

### 2. K-Nearest Neighbors (KNN)
- **Tahapan**:
  KNN memprediksi kelas berdasarkan kelas mayoritas dari K tetangga terdekat.
  - **Parameter utama**: K (jumlah tetangga) dan metric (jarak Euclidean).
  - **Scaling data**: Sama seperti SVC, KNN sensitif terhadap skala fitur.
- **Kelebihan**: 
  - Mudah diimplementasikan dan dipahami.
  - Cocok untuk dataset kecil.
- **Kekurangan**: 
  - Performa menurun jika jumlah data sangat besar karena KNN harus menghitung jarak untuk setiap data baru.

### 3. Decision Tree Classifier
- **Tahapan**:
  Decision Tree membagi data berdasarkan pemisahan yang memaksimalkan "information gain".
  - **Parameter utama**: Maximum depth (kedalaman maksimum pohon) dan criterion (Gini atau Entropy).
- **Kelebihan**:
  - Mudah dipahami dan divisualisasikan.
  - Tidak membutuhkan scaling data.
- **Kekurangan**: 
  - Rentan terhadap overfitting jika tidak dilakukan pruning pada pohon yang terlalu dalam.

### Pemilihan Model Terbaik
<img width="233" alt="hasil_model" src="https://github.com/user-attachments/assets/1483f9bf-f361-4f90-8c57-565a7c778658">

Setelah melatih ketiga model, model **Support Vector Classifier (SVC)** dipilih sebagai model terbaik karena memiliki akurasi tertinggi pada set pengujian, yaitu **98.36%**. Model ini dipilih karena mampu memisahkan kelas dengan sangat baik dan performa yang optimal pada data ini.
Untuk membuktikannya, model **Support Vector Classifier (SVC)** diuji pada data uji dan di visualisasikan pada confussion matrix seperti berikut.

![predicted](https://github.com/user-attachments/assets/e945f986-f3b1-4af9-b92b-713b1fc54845)

Visualisasi di atas menunjukkan bahwa bagian kiri atas mewakili TN (True Negative), yaitu data negatif yang diprediksi dengan benar, sedangkan bagian kanan bawah menunjukkan data positif yang diprediksi dengan benar. Sebaliknya, bagian kanan atas menampilkan data False Negative, yaitu data positif yang salah diprediksi sebagai negatif, dan bagian kiri bawah menampilkan False Positive, yaitu data negatif yang salah diprediksi sebagai positif.

## Evaluation

Pada proyek ini, model yang dikembangkan berfokus pada kasus klasifikasi dan menggunakan metrik evaluasi yang mencakup **akurasi**, **F1-score**, **recall**, dan **precision**. Hasil pengukuran performa model terbaik yang menggunakan algoritma **Support Vector Classifier (SVC)** dapat dilihat pada tabel di bawah ini:

<img width="317" alt="SupportVectorClassifier" src="https://github.com/user-attachments/assets/f4c0a699-ffe9-4eb1-a35c-643b0a99e829">

### Pejelasan Metrik

<img width="122" alt="metric" src="https://github.com/user-attachments/assets/72c9c8b3-768f-46f7-b5e7-0a7441810b7e">

### Akurasi
Akurasi adalah metrik yang digunakan untuk mengukur proporsi total data yang berhasil diprediksi dengan benar oleh model.

- **True Positive (TP)**: Kasus di mana model berhasil memprediksi data positif dengan benar. Contohnya, pasien yang menderita kanker (kelas 1) dan model memprediksi pasien tersebut benar sebagai kanker (kelas 1).
- **True Negative (TN)**: Kasus di mana model berhasil memprediksi data negatif dengan benar. Misalnya, pasien yang tidak menderita kanker (kelas 2) dan model memprediksi pasien tersebut tidak menderita kanker (kelas 2).
- **False Positive (FP) - Type I Error**: Kasus di mana model salah memprediksi data negatif sebagai positif. Contohnya, pasien yang tidak menderita kanker (kelas 2), tetapi model memprediksi bahwa pasien tersebut menderita kanker (kelas 1).
- **False Negative (FN) - Type II Error**: Kasus di mana model salah memprediksi data positif sebagai negatif. Misalnya, pasien yang sebenarnya menderita kanker (kelas 1), tetapi model memprediksi bahwa pasien tersebut tidak menderita kanker (kelas 2).

### Precision
*Precision* mengukur kemampuan model dalam mengklasifikasikan data positif dengan benar dari seluruh prediksi positif yang dihasilkan.
Precision penting dalam konteks medis untuk memastikan bahwa ketika model memprediksi pasien memiliki kanker, sebagian besar prediksi tersebut memang benar.

### Recall
*Recall* mengukur seberapa baik model dapat mendeteksi semua contoh positif yang sebenarnya dari seluruh data positif.
Metrik ini sangat penting untuk memastikan bahwa semua pasien yang menderita kanker terdeteksi oleh model, sehingga tidak ada kasus yang terlewat.

### F1-Score
*F1-score* adalah metrik yang menggabungkan precision dan recall dengan memberikan rata-rata harmonik di antara keduanya. Ini berguna ketika kita ingin mencari keseimbangan antara precision dan recall.
F1-score menjadi sangat relevan dalam konteks ketidakseimbangan kelas, di mana kita mungkin memiliki lebih banyak kasus negatif daripada positif.

### Hasil Evaluasi
Hasil evaluasi menunjukkan bahwa model **Support Vector Classifier (SVC)** berhasil mencapai akurasi sebesar **98.36%**. Model ini juga menunjukkan nilai precision dan recall yang tinggi, yang mencerminkan kemampuannya dalam mendeteksi tumor ganas (kanker) dengan akurasi yang baik. Dengan hasil F1-score yang tinggi, model ini menunjukkan keseimbangan yang baik antara keakuratan dalam klasifikasi positif dan kemampuan untuk mendeteksi semua contoh positif.

Secara keseluruhan, metrik evaluasi ini menunjukkan bahwa model yang digunakan efektif dalam mendeteksi kanker payudara, memberikan keyakinan bahwa pasien yang membutuhkan perawatan dapat diidentifikasi dengan baik melalui sistem yang dikembangkan.


---Ini adalah bagian akhir laporan---
