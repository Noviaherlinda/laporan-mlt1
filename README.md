# laporan-mlt1 NOVIA HERLINDA MARIUS
m496y1045
M04


## Project Overview
Perusahaan yang bergerak di bidang kesehatan membutuhkan sistem yang dapat melakukan diagnosa penyakit diabetes dengan baik. Berdasarkan data dari dinas kesehatan 
Data Riskesdas 2018 menunjukkan jumlah keseluruhan kasus penyakit diabetes yang ada di Indonesia yakni sebesar 8,5%, meningkat dibandingkan Riskesdas 2013 yaitu sebesar 6,9%. (https://dinkes.kalbarprov.go.id/diabetes-sebabkan-kematian-tertinggi-di-indonesia-atasi-secepatnya-sebelum-terlambat/)

irwansyah, dkk melakukan penelitian untuk melakukan diagnosa penyakit diabetes menggunakan algoritma ADAP menghasilkan skor *sensitivity* dan *specifity* sebesar 0.76
[[1]](file:///C:/Users/ASUS/Downloads/511-Research%20Articles-3484-4-10-20210527.pdf)
Algoritma ADAP merupakan salah satu metode *Machine Learning*. Sejalan dengan penelitian tersebut, solusi yang ditawarkan yaitu menggunakan pendekatan *Machine Learning* dengan metode KNN, Random Forest dan AdaBoosting untuk melakukan diagnosis penyakit diabetes (klasifikasi apakah pasien mengidap diabetes atau tidak).

## Business Understanding

### Problem Statements
- Bagaimana cara membangun sistem prediksi untuk melakukan diagnosa penyakit diabetes dengan model terbaik?
### Goals

- Bagaimana cara membangun sistem prediksi penyakit diabetes berdasarkan dengan model terbaik?

## Data Understanding

Tabel 1. Informasi Dataset\
| | Keterangan |
|---|---|
| Sumber | [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| Jumlah Data | 768 |
| *Usability* | 8.82 |
| Lisensi | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) |
| *Rating* | *gold* |
| Jenis dan Ukuran Berkas | csv (9 kB) |

### Variabel-variabel pada Dataset

Berdasarkan informasi dari [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), variabel-variabel pada Diabetes dataset adalah sebagai berikut:

- Pregnancies: merepresentasikan berapa kali hamil
- Glucose: merepresentasikan konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral
- BloodPressure: merepresentasikan tekanan darah diastolik (mm Hg)
- SkinThickness: merepresentasikan ketebalan lipatan kulit trisep (mm)
- Insulin: merepresentasikan insulin serum 2 jam (mu U/ml)
- BMI: merepresentasikan indeks massa tubuh (berat dalam kg/(tinggi dalam m)$^2$)
- DiabetesPedigreeFunction: merepresentasikan fungsi silsilah diabetes
- Age: merepresentasikan umur pengguna
- Outcome: merepresentasikan status diagnosa pengguna apakah terdiagnosa diabetes (1) atau tidak (0).

### Menangani Missing Value

Untuk mendeteksi *missing value* digunakan fungsi isnull().sum() dan diperoleh:

Tabel 2. Hasil Deteksi *Missing Value*

| Kolom | Jumlah *Missing Value* |
|---|:---:|
| Pregnancies | 0 |
| Glucose | 0 |
| BloodPressure | 0 |
| SkinThickness | 0 |
| Insulin | 0 |
| BMI | 0 |
| DiabetesPedigreeFunction | 0 |
| Age | 0 |
| Outcome | 0 |

Dari Tabel 2 di atas, terlihat bahwa setiap fitur tidak memiliki *missing value*.

### Univariate Analysis

Selanjutnya, untuk fitur numerik, akan dilakukan visualisasi dengan histogram pada masing-masing fiturnya sebagai berikut.

![histogram](https://user-images.githubusercontent.com/119787827/205722685-5cd65195-015e-49e9-9f98-ccd3d7a31877.png)


Gambar 1. Histogram pada Setiap Fitur Numerik

Berdasarkan Gambar 1. di atas, diperoleh beberapa informasi, antara lain:
- Pada histogram DiabetesPedigreeFunction dan histogram Age miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model.

# Multivariate Analysis

Untuk mengamati hubungan antara fitur numerik, akan digunakan fungsi pairplot(), dengan output sebagai berikut.

![multivariate analysis](https://user-images.githubusercontent.com/119787827/205723139-7d37b23f-3c7b-4a16-9f83-c87bcd1e66ce.png)


Gambar 2. Visualisasi Hubungan antar Fitur Numerik

Pada pola sebaran data grafik pairplot di atas, terlihat fitur Age memiliki korelasi cukup kuat (positif) dengan fitur Pregnancies. Untuk mengevaluasi skor korelasinya, akan digunakan fungsi corr() sebagai berikut.

![correlation](https://user-images.githubusercontent.com/119787827/205723253-c6706270-6a8a-4017-ab7e-135fafab1d4c.png)


Gambar 3. Korelasi antar Fitur Numerik

Koefisien korelasi berkisar antara -1 dan +1. Semakin dekat nilainya ke 1 atau -1, maka korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0 maka korelasinya semakin lemah.

Dari grafik korelasi di atas, fitur Age memiliki korelasi yang cukup kuat (0.54) dengan fitur target Pregnancies.

## Data Preparation

### Reduksi Dimensi dengan PCA

PCA umumnya digunakan ketika variabel dalam data yang memiliki korelasi yang tinggi. Korelasi tinggi ini menunjukkan data yang berulang atau redundant. Sebelumnya perlu cek kembali korelasi antar fitur (selain fitur target) dengan menggunakan pairplot.

![PCA](https://user-images.githubusercontent.com/119787827/205727155-ac1d218c-ccce-4915-930e-8a5f01753265.png)


Gambar 4. Visualisasi Hubungan antar Fitur Selain Fitur Target (Outcome)

Selanjutnya kita akan mereduksi fitur Age dan fitur Pregnancies karena keduanya berkorelasi cukup kuat yang dapat dilihat pada visualisasi pairplot di atas.

Untuk implementasinya menggunakan fungsi PCA() dari sklearn dengan mengatur nilai parameter n_components sebanyak fitur yang akan dikenakan PCA.

Tabel 3. Proporsi *Principal Component* dari Hasil PCA

| PC Pertama | PC Kedua |
|:---:|:---:|
| 0.948 | 0.052 |

Arti dari output di atas adalah, 94.8% informasi pada kedua fitur (Age dan Pregnancies) terdapat pada PC (Principal Component) pertama. Sedangkan sisanya sebesar 5.2% terdapat pada PC kedua

Berdasarkan hasil tersebut, kita akan mereduksi fitur dan hanya mempertahankan PC (komponen) pertama saja. PC pertama ini akan menjadi fitur yang menggantikan dua fitur lainnya (Age dan Pregnancies). Kita beri nama fitur ini PCA_1

Tabel 4. Tampilan 5 Sampel dari Dataset Setelah Dilakukan Reduksi Fitur

|index|Glucose|BloodPressure|SkinThickness|Insulin|BMI|DiabetesPedigreeFunction|Outcome|PCA\_1|
|---|---|---|---|---|---|---|---|---|
|120|162|76|56|100|53\.2|0\.759|1|-8\.757570419483171|
|596|67|76|0|0|45\.3|0\.194|0|11\.961361072477404|
|677|93|60|0|0|35\.3|0\.263|0|-8\.757570419483171|
|236|181|84|21|192|35\.9|0\.586|1|18\.03587563665654|
|58|146|82|0|0|40\.5|1\.781|0|9\.988129501814493|

### Train Test Split

Pada tahap ini akan dibagi dataset menjadi data latih (train) dan data uji (test). Pada kasus ini akan menggunakan proporsi pembagian sebesar 80:20 dengan fungsi train_test_split dari sklearn.

Tabel 5. Jumlah Data Latih dan Uji

| Jumlah Data Latih | Jumlah Data Uji | Jumlah Total Data |
|:---:|:---:|:---:|
| 614 | 154 | 768 |

### Standarisasi

Proses standarisasi bertujuan untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn.

StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandarScaler menghasilkan distribusi deviasi sama dengan 1 dan mean sama dengan 0.

Tabel 6. Hasil Proses Standarisasi pada Setiap Fitur

|index|Glucose|BloodPressure|SkinThickness|Insulin|BMI|DiabetesPedigreeFunction|PCA\_1|
|---|---|---|---|---|---|---|---|
|318|-0\.19254286227071132|-0\.14765856099994998|1\.1603915557171882|0\.5329581162164486|0\.7709162511660301|-0\.9608240948723352|-0\.444504589562716|
|313|-0\.2562396927848367|-0\.9709262700030774|-0\.6682598329986497|0\.04416108233280961|-0\.3589991380301563|0\.5132898978087475|-0\.6953935511276422|
|195|1\.1769389937829844|0\.7785176116285684|1\.2865054445941426|1\.1550634320683528|0\.9417174146491742|-0\.20208895158060144|-0\.3332312705986419|
|570|-1\.3709342267820308|0\.058158366250831865|-1\.2988292773834214|-0\.7112525154873598|0\.0351573930847924|-0\.5891987185661799|0\.4754216028420135|
|226|-0\.638420675869589|0\.3668837571270046|-1\.2988292773834214|-0\.7112525154873598|0\.4555910262740714|-0\.812173944349873|-0\.6532293949363148|

## Modeling
Pada tahap ini, kita akan menggunakan tiga algoritma untuk kasus klasifikasi ini. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menetukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan kita gunakan, antara lain:

1. K-Nearest Neighbor

    Kelebihan algoritma KNN adalah mudah dipahami dan digunakan sedangkan kekurangannya kika dihadapkan pada jumlah fitur atau dimensi yang besar rawan terjadi bias.

2. Random Forest
    
    Kelebihan algoritma Random Forest adalah menggunakan teknik Bagging yang berusaha melawan *overfitting* dengan berjalan secara paralel. Sedangkan kekurangannya ada pada kompleksitas algoritma Random Forest yang membutuhkan waktu relatif lebih lama dan daya komputasi yang lebih tinggi dibanding algoritma seperti Decision Tree.

3. Boosting Algorithm

    Kelebihan algoritma Boosting adalah menggunakan teknik Boosting yang berusaha menurunkan bias dengan berjalan secara sekuensial (memperbaiki model di tiap tahapnya). Sedangkan kekurangannya hampir sama dengan algoritma Random Forest dari segi kompleksitas komputasi yang menjadikan waktu pelatihan relatif lebih lama, selain itu noisy dan outliers sangat berpengaruh dalam algoritma ini.

Untuk langkah pertama, kita akan siapkan DataFrame baru untuk menampung nilai metrik Akurasi pada setiap model / algoritma. Hal ini berguna untuk melakukan analisa perbandingan antar model.

### Model KNN
KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih k tetangga terdekat. Pemilihan nilai k sangat penting dan berpengaruh terhadap performa model. Jika kita memilih k yang terlalu rendah, maka akan menghasilkan model yang *overfitting* dan hasil prediksinya memiliki varians tinggi. Jika kita memilih k yang terlalu tinggi, maka model yang dihasilkan akan *underfitting* dan prediksinya memiliki bias yang tinggi [[3]](https://learning.oreilly.com/library/view/machine-learning-with/9781617296574/).

Oleh karena itu, kita akan mencoba beberapa nilai k yang berbeda (1 sampai 20) kemudian membandingan mana yang menghasilkan nilai metrik model (pada kasus ini kita pakai akurasi) terbaik. Selain itu, kita akan menggunakan metrik ukuran jarak secara default (Minkowski Distance) pada *library* sklearn.

Tabel 7. Perbandingan Nilai K terhadap Akurasi

| K | Akurasi |
|:---:|---|
| 1 | 0.7142857142857143 |
| 2 | 0.6948051948051948 |
| 3 | 0.7272727272727273 |
| 4 | 0.7207792207792207 |
| 5 | 0.7597402597402597 |
| 6 | 0.7662337662337663 |
| 7 | 0.7857142857142857 |
| 8 | 0.7727272727272727 |
| 9 | 0.7662337662337663 |
| 10 | 0.7532467532467533 |
| 11 | 0.7467532467532467 |
| 12 | 0.7597402597402597 |
| 13 | 0.7467532467532467 |
| 14 | 0.7532467532467533 |
| 15 | 0.7532467532467533 |
| 16 | 0.7467532467532467 |
| 17 | 0.7467532467532467 |
| 18 | 0.7272727272727273 |
| 19 | 0.7467532467532467|
| 20 | 0.7662337662337663 |

Jika divisualisasikan dengan fungsi `plot()` diperoleh:

![visualisasi](https://user-images.githubusercontent.com/119787827/205727311-f9d4a937-48bd-432d-ad2c-c02f450eaa0a.png)


Gambar 5. Visualisai Nilai K terhadap Akurasi

Dari hasil output diatas, nilai akurasi terbaik dicapai ketika k = 7 yaitu sebesar 0.7857. Oleh karena itu kita akan menggunakan k = 7 dan menyimpan nilai akurasi nya (terhadap data latih, untuk data uji akan dilakukan pada proses evaluasi) kedalam df_models yang telah kita siapkan sebelumnya.

## Model Random Forest

Random forest merupakan algoritma *supervised learning* yang termasuk ke dalam kategori *ensemble* (group) learning. Pada model *ensemble*, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model *ensemble* ini digabungkan untuk membuat prediksi akhir. Jenis metode *ensemble* yang digunakan pada Random Forest adalah teknik Bagging. Metode ini bekerja dengan membuat subset dari data train yang independen. Beberapa model awal (base model / weak model) dibuat untuk dijalankan secara simultan / paralel dan independen satu sama lain dengan subset data train yang independen. Hasil prediksi setiap model kemudian dikombinasikan untuk menentukan hasil prediksi final. 

Parameter-parameter (*hyperparameter*) yang digunakan pada algoritma ini antara lain:

- n_estimator: jumlah trees (pohon) di forest.
- max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.

Untuk menentukan nilai *hyperparameter* (n_estimator & max_depth), dilakukan tuning dengan GridSearchCV dan hasilnya sebagai berikut:

Tabel 8. Hasil *Hyperparameter Tuning* model *GridSearchCV* dengan Random Forest

| | Daftar Nilai | Nilai Terbaik |
|---|---|---|
| n_estimators | 10, 20, 30, 40, 50, 60, 70, 80, 90 | 70 |
| max_depth | 4, 8, 16, 32 | 16 |
| Accuracy data latih | | 1.0 |
| Accuracy data uji | | 0.773 |

Dari hasil output di atas diperoleh nilai Akurasi terbaik dalam jangkauan parameter params_rf yaitu 1.0 (dengan data train) dan 0.7727 (dengan data test) dengan n_estimators: 70 dan max_depth: 16. Selanjutnya kita akan menggunakan pengaturan parameter tersebut dan menyimpan nilai Akurasi nya kedalam df_models yang telah kita siapkan sebelumnya.

### Model AdaBoosting

Jika sebelumnya kita menggunakan algoritma *bagging* (Random Forest). Selanjutnya kita akan menggunakan metode lain dalam model *ensemble* yaitu teknik *Boosting*. Algoritma *Boosting* bekerja dengan membangun model dari data train. Kemudian membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Teknik ini bekerja secara sekuensial.

Pada kasus ini kita akan menggunakan metode *Adaptive Boosting*. Untuk implementasinya kita menggunakan AdaBoostClassifier dari library sklearn dengan base_estimator defaultnya yaitu DecisionTreeClassifier hampir sama dengan RandomForestClassifier bedanya menggunakan metode teknik *Boosting*.

Parameter-parameter (hyperparameter) yang digunakan pada algoritma ini antara lain:

- n_estimator: jumlah *estimator* dan ketika mencapai nilai jumlah tersebut algoritma Boosting akan dihentikan.
- learning_rate: bobot yang diterapkan pada setiap *classifier* di masing-masing iterasi Boosting.
- random_state: digunakan untuk mengontrol *random number* generator yang digunakan.

Untuk menentukan nilai *hyperparameter* (n_estimator & learning_rate) di atas, kita akan melakukan *tuning* dengan GridSearchCV.

Tabel 9. Hasil *Hyperparameter Tuning* model *GridSearchCV* dengan AdaBoosting

| | Daftar Nilai | Nilai Terbaik |
|---|---|---|
| n_estimators | 10, 20, 30, 40, 50, 60, 70, 80, 90 | 90 |
| learning_rate | 0.001, 0.01, 0.1, 0.2 | 0.2 |
| Accuracy data latih | | 0.8094 |
| Accuracy data uji | | 0.7402 |

Dari hasil output di atas diperoleh nilai Akurasi terbaik dalam jangkauan parameter params_ab yaitu  0.8094 (dengan data train) dan 0.7402 (dengan data test) dengan n_estimators: 90 dan learning_rate: 0.2. Selanjutnya kita akan menggunakan pengaturan parameter tersebut dan menyimpan nilai Akurasi nya kedalam df_models yang telah kita siapkan sebelumnya.

## Evaluation
Dari proses sebelumnya, telah dibangun dan dilatih tiga model yang berbeda (KNN, Random Forest, Boosting). Selanjutnya perlu mengevaluasi model-model tersebut menggunakan data uji dan metrik yang digunakan dalam kasus ini yaitu akurasi. Hasil evaluasi kemudian disimpan ke dalam df_models.

$$\texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)$$

Dengan:
- $n_{\text{sample}}$ adalah banyaknya data
- $1(\hat{y}_i = y_i)$ bernilai 1 jika $\hat{y}_i$ nilainya sama dengan $y_i$. Dimana $\hat{y}_i$ adalah hasil prediksi sedangkan $y_i$ adalah nilai yang akan diprediksi (nilai yang sebenarnya).

Berdasarkan DataFrame `df_models` diperoleh:

Tabel 10. Nilai Akurasi pada Setiap Model dengan Data Uji

|index|KNN|RandomForest|Boosting|
|---|---|---|---|
|Train Accuracy|0\.8078175895765473|0\.9543973941368078|0\.7899022801302932|
|Test Accuracy|0\.7857142857142857|0\.7792207792207793|0\.7727272727272727|

Untuk memudahkan, dilakukan *plot* hasil evaluasi model dengan *bar chart* sebagai berikut:

![evaluasi](https://user-images.githubusercontent.com/119787827/205727458-66c457c8-10e9-4bc6-a7be-3cd080e91ab0.png)


Gambar 6. *Bar Chart* Hasil Evaluasi Model dengan Data Latih dan Uji

Dari gambar di atas, terlihat bahwa, model RandomForest memberikan nilai Akurasi (pada data uji) yang paling tinggi. Sebelum memutuskan model terbaik untuk melakukan prediksi "Outcome" atau hasil diagnosa terhadap penyakit diabetes. Mari kita coba uji prediksi menggunakan beberapa sampel acak (10) pada data uji.

Tabel 11. Hasil Prediksi dari 10 Sampel Acak

|index\_sample|y\_true|prediksi\_KNN|prediksi\_RF|prediksi\_Boosting|
|---|---|---|---|---|
|64|1|1|1|0|
|9|1|0|0|0|
|363|1|0|1|0|
|697|0|0|0|0|
|200|0|0|0|0|
|41|0|1|1|1|
|171|1|0|1|1|
|242|1|0|0|0|
|107|0|1|0|1|
|11|1|1|1|1|

Dari Tabel 11, terlihat bahwa prediksi dengan Random Forest memberikan hasil yang paling mendekati.

## Conclusion
Berdasarkan hasil evaluasi model di atas, dapat disimpulkan bahwa model terbaik untuk melakukan klasifikasi "Outcome" atau diagnosa penyakit diabetes adalah model Random Forest.
