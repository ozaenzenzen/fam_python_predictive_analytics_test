# Comparison KNN, Random Forest, and Boosting Algorithm algorithm of Customer Churn Dataset  - Fauzan Akmal Mahdi

## Domain Proyek
***
### Latar Belakang

Dalam beberapa bidang kehidupan, khususnya dalam perusahaan bidang keuangan / finansial, data mengenai penghasilan individu merupakan sebuah hal yang dibutuhkan. Beberapa perusahaan dalam membutuhkan data mengenai penghasilan untuk memetakan karakter dan kemampuan individu, sebagai contoh pada bidang Human Resources untuk memetakan kemampuan dan rentang penghasilan pada individu yang melamar kerja. Pada contoh lain yaitu Bank, data pengahasilan individu digunakan untuk memetakan kemampuan bayar dari individu. Dengan bantuan teknologi, lebih khususnya penggunaan Machine Learning, penarikan informasi dari sebuah data menjadi lebih efektif dan efisien. Pada penelitian ini ditujukan membandingkan kemampuan metode machine learning dalam memetakan estimasi pendapatan (estimated salary) dari data mengenai kemampuan bayar individu. Selanjutnya, metode terbaik dari penelitian ini dapat digunakan untuk melakukan pemodelan terhadap data penghasilan yang lain.

Dataset merupakan koleksi data yang berfokus kepada prediksi customer churn. Dataset berisi berbagai fitur yang menggambarkan setiap customer, seperti  credit score, country, gender, age, tenure, balance, number of products, credit card status, active membership, estimated salary, and churn status. Status churn menunjukkan apakah pelanggan telah melakukan churn atau belum. Kumpulan data ini digunakan untuk menganalisis dan memahami faktor-faktor yang berkontribusi terhadap churn pelanggan dan untuk membangun model prediktif guna mengidentifikasi pelanggan yang berisiko churn. Tujuannya adalah untuk melakukan estimasi salary terhadap customer guna menemukan rentang salary dari setiap customer.

*XXTujuannya adalah untuk mengembangkan strategi dan intervensi untuk mengurangi churn dan meningkatkan retensi pelangganXX* 

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana langkah dalam mempersiapkan data untuk dilakukan pelatihan model machine learning?
- Bagaimana konfigurasi metode machine learning yang digunakan?
- Bagaimana kemampuan metode machine learning yang digunakan?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengetahui langkah dalam mempersiapkan data untuk dilakukan pelatihan model machine learning?
- Mengetahui konfigurasi metode machine learning yang digunakan
- Mengetahui kemampuan metode machine learning yang digunakan

### Solution statements
- Menggunakan model machine learning K-Nearest Neighbors (KNN), Random Forest, Boosting Algorithm, dan Linear Regression 
- Membuat model machine learning
- Menggunakan metode evaluasi Mean Squared Error dan R2 Score

## Data Understanding
Data yang digunakan bersumber dari [kaggle](https://www.kaggle.com/datasets/bhuviranga/customer-churn-data/data)

Spesifikasi dataset yang digunakan pada Table 1

Table 1. Informasi Dataset
| Jenis | Keterangan |
| --- | --- |
| Sumber Dataset | [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/bhuviranga/customer-churn-data/data) |
| Kategori Dataset | Open Dataset |
| Lisensi Dataset | [License](https://opendatacommons.org/licenses/dbcl/1-0/) |
| Jenis Dataset | Comma-Separated Values  (CSV) |
| Ukuran Dataset |  561.6 kB |

Dataset berisi 12 kolom yaitu **credit score, country, gender, age, tenure, balance, number of products, credit card status, active membership, estimated salary, and churn status.** dan 10000 baris. Dengan 2 variabel bersifat kategorikal dan 10 variabel bersifat numerik. Berikut penjelasan lebih lengkap terhadap variabel dalam dataset

### Variabel-variabel pada Bank Customer Churn Dataset adalah sebagai berikut:
- credit score: merupakan angka skor kredit (peminjaman) dari customer bank
- country: merupakan negara asal dari customer bank (kategorik)
- gender: merupakan jenis kelamin customer bank (kategorik)
- age: merupakan usia dari customer bank (kategorik)
- tenure: merupakan angka tenor dari customer bank
- balance: merupakan angka tabungan dari customer bank
- number of products: merupakan angka jenis produk yang digunakan oleh customer bank
- credit card status: merupakan angka status kartu kredit dari customer bank
- active membership: merupakan angka status keatifkan anggota dari customer bank
- estimated salary: merupakan angka estimasi salary dari customer bank
- churn status: merupakan angka status churn (kehilangan keanggotaan) dari customer bank

## Data Preparation
Langkah dalam persiapan data dijabarkan dalam beberapa poin
1. Mendownload dataset dari kaggle
    - Proses: Mengambil dataset dari website kaggle 
    - Alasan: Dataset akan digunakan untuk pengolahan dan pemodelan pada penelitian
2. Menyimpan dataset ke dalam Google Drive
    - Proses: Menyimpan dataset ke dalam penyimpanan cloud Google Drive
    - Alasan: Penyimpanan dataset dalam Google Drive agar mempermudah proses pemanggilan data dalam proses penelitian
3. Melakukan pemanggilan library pendukung
    - Proses: Melakukan pengimporan beberapa library dari numpy, matplotlib, pandas, dan seaborn dengan `import`
    - Alasan: Ditujukan untuk mendukung proses pengolahan, pelatihan, dan evaluasi data seperti pengolahan, visualisasi, dan proses matematis
4. Melakukan mounting Google Colab dengan Google Drive 
    - Proses: Melakukan koneksi antar platform Google Colab sebagai media pengolahan data dan Google Drive sebagai media penyimpanan data. Menggunakan library `from google.colab import drive` dan menggunakan kode `drive.mount('/content/drive')`
    - Alasan: Agar proses persiapan data dan pengolahan model bisa dilakukan dari data yang disimpan dalam Google Drive dan media pengolahan data di Google Colab
5. Memanggil dataset 
    - Proses: Melakukan pemanggilan dataset dengan library pandas `pd.read_csv(url)`
    - Alasan: Agar dataset siap digunakan pada proses pengolahan dalam Google Colab
6. Melakukan pengecekan informasi dataset
    - Proses: Pengecekan informasi dataset terbagi menjadi sub tahap seperti
        - menggunakan `dataset.info()` untuk menampilkan jumlah baris, missing value, dan tipe data dari dataset
        - menggunakan `dataset.describe()` untuk mendeskripsikan parameter singkat pada dataset
            - Count atau jumlah baris dari setiap kolom
            - Mean atau rata-rata dari setiap kolom
            - Standar deviasi dari setiap kolom
            - Nilai minimum / terkecil dari setiap kolom
            - Nilai kuartil pertama atau 25% dari setiap kolom
            - Nilai kuartil kedua atau 50% atau median dari setiap kolom
            - Nilai kuartil ketiga atau 75% dari setiap kolom
            - Nilai maximum / terbesar dari setiap kolom
        - menggunakan `dataset.column` untuk mengetahui kolom yang tersedia pada dataset
    - Alasan: Untuk mengetahui informasi dasar dari dataset yang akan digunakan sehingga dapat ditentukan langkah selanjutnya dalam melakukan persiapan data
7. Melakukan pengecekan missing value dalam data (bisa NULL atau 0) 
    - Proses: Menggunakan `dataset.isnull().sum()*100/dataset.shape[0]` untuk mendapatkan informasi apakah terdapat data yang bersifat null
    - Alasan: Untuk mengetahui informasi missing value dari dataset yang akan digunakan sehingga dapat ditentukan langkah selanjutnya dalam melakukan persiapan data
8. Melakukan outlier analisis & memvisualisasikan persebaran data pada setiap kolom untuk mengetahui outlier = sebuah data atau observasi yang menyimpang secara ekstrim dari rata-rata sekumpulan data yang ada
    - Proses: Menggunakan fungsi `boxplot()` dari seaborn untuk mendapatkan informasi visualisasi persebaran data dari setiap kolom
    - Alasan: Untuk mengetahui informasi persebaran nilai apakah terdapat nilai outlier dari dataset yang akan digunakan sehingga dapat ditentukan langkah selanjutnya dalam melakukan persiapan data
9. Drop Outlier yaitu menangani outlier pada dataset. Disini digunakan metode Inter Quartile Range (IQR)
    - Proses: Menggunakan metode Inter Quartile Range (IQR) untuk menghilangkan nilai outlier pada dataset sehingga didapatkan informasi dataset 7677 baris dan 12 kolom
    - Alasan: Proses penghilangan outlier dilakukan karena terdapat nilai outlier pada kolom **credit_score** dan **age**
10. Melakukan univariate analysis yaitu mengeksplorasi dan menjelaskan setiap variabel dalam kumpulan data secara terpisah untuk 1 jenis variabel / kolom
    - Proses: Proses dalam tahap ini ditujukan untuk mengeksplorasi dan menjelaskan setiap variabel dalam kumpulan data secara terpisah untuk 1 jenis variabel / kolom. Tahap ini terbagi menjadi sub tahap seperti berikut
        - Membagi kolom yang bersifat numerk dan kategorikal 
        - Menyimpan kolom ke dalam masing-masing variabel yaitu `numerical_features` dan `categorical_features`
        - Melakukan visualisasi untuk menginformasikan data pada kolom kategorik yaitu country dan gender
            - Didapatkan hasil pada kolom `country` pada Table 2
            - Table 2. hasil univariate analysis pada kolom kategorik `country`
            <table><tbody><tr><td>Data</td><td>Jumlah Sampel</td><td>Persentase</td></tr><tr><td>France</td><td>4049</td><td>52.7</td></tr><tr><td>Spain</td><td>1988</td><td>25.9</td></tr><tr><td>Germany</td><td>1640</td><td>21.4</td></tr></tbody></table>
            - Didapatkan hasil pada kolom `gender` pada Table 3
            - Table 3. hasil univariate analysis pada kolom kategorik `gender`
            <table><tbody><tr><td>Data</td><td>Jumlah Sampel</td><td>Persentase</td></tr><tr><td>Male</td><td>4399</td><td>57.3</td></tr><tr><td>Female</td><td>3278</td><td>42.7</td></tr></tbody></table>
        - Melakukan visualisasi persebaran nilai dalam bentuk grafik untuk menginformasikan data pada kolom numerik yaitu 'customer_id', 'credit_score', 'age', 'tenure',
       'balance', 'products_number', 'credit_card', 'active_member',
       'estimated_salary', 'churn'
    - Alasan: Untuk mengetahui informasi persebaran nilai pada kolom kateogirkal dan numerikal secara spesifik univariate per kolom pada dataset yang digunakan sehingga dapat ditentukan langkah selanjutnya dalam melakukan persiapan data
11. Melakukan multivariate analysis yaitu mengeksplorasi dan menjelaskan setiap variabel dalam kumpulan data secara terpisah untuk 2 atau lebih jenis variabel / kolom
    - Proses: Pada tahap ini terbagi menjadi beberapa sub tahap sebagai berikut
        - Visualisasi data kategorik `country` dan `gender` terhadap data numerik yang dipilih `balance`, `estimated_salary`, `credit_score`, dan `age` menggunakan fungsi `pairplot()`
        - Visualisasi antar data kolom numerik menggunakan fungsi `pairplot()`
        - Visualisasi matriks korelasi antar data kolom numerik untuk mendapatkan nilai koefisien korelasi
    - Alasan: Untuk mengetahui informasi persebaran nilai antar kolom kateogirkal dan numerikal dan kolom numerik dengan kolom numerik lainnya  pada dataset yang digunakan sehingga dapat ditentukan langkah selanjutnya dalam melakukan persiapan data. Didapatkan korelasi negatif tertinggi pada kolom `balance` dan `products_number` yaitu -0,42
12. Melakukan penghilangan kolom yang tidak diperlukan sesuai analisis masalah dan tujuan penelitian
    - Proses: Menghilangkan kolom fitur yang tidak diperlukan yaitu `customer_id` menggunakan fungsi `drop()`
    - Alasan: Korelasi kolom `customer_id` terhadap dataset sangat kecil karena secara sifatnya yang hanya sebuah identifier unik pada baris kolom
13. Melakukan Encoding Categorical Features yaitu memberikan alias dalam bentuk numerik kepada kolom yang bersifat kategorikal
    - Proses: Melakukan pemberian alias terhadap kolom kategorik agar bisa dapat berbentuk numerik dan memisahkan data kategorik sebagai kolom terpisah
    - Alasan: Pemberian alias terhadap kolom kategorik `country` dan `gender` ditujukan agar kolom fitur tersebut dapat digunakan dalam proses pemodelan
14. Melakukan PCA Reduction yaitu mereduksi dimensi, mengekstraksi fitur, dan mentransformasi data dari “n-dimensional space” ke dalam sistem berkoordinat baru dengan dimensi m, di mana m lebih kecil dari n
    - Proses: Melakukan proses reduksi dimensi kolom dataset dengan menggunakan metode PCA. Kolom yang dilakukan reduksi PCA adalah `credit_card','active_member','churn'`
    - Alasan: Secara numerik, kolom `credit_card','active_member','churn'` memiliki rentang nilai antara 0 sampai 1 sehingga dapat direduksi menjadi 1 kolom tanpa mengurangi kualitas fitur data
15. Melakukan pembagian dataset menjadi data train dan data test dalam pembagian yang ditentukan
    - Proses: Melakukan pembagian data test dan data train dengan pembagian 10% data test dan 90% data train menggunakan fungsi `train_test_split(X, y, test_size = 0.1, random_state = 123)`
    - Alasan: Pembagian dataset disesuaikan agar tidak terjadi overfit pada proses pemodelan yang akan dilakukan dengan metode machine learning
16. Melakukan standarisasi atau perubahan skala nilai pada suatu kolom sesuai skala yang diinginkan. Fitur kolom yang menjadi tujuan adalah kolom `estimated_salary`
    - Proses: Melakukan standarisasi nilai pada kolom `balance` dengan menggunakan funsi dari sklearn yaitu `StandardScaler`
    - Alasan: Proses standarisasi data dilakukan agar rentang nilai pada kolom `balance` tidak terlampau jauh dan agar menyelaraskan dengan kolom dataset lain

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

