# Data Sciene Edutech - Andrian Syah

# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan
 
## Business Understanding
Jaya Jaya Institut adalah institusi pendidikan tinggi yang telah memasuki tahun ke-22 sejak didirikan pada tahun 2000. Institusi ini telah berhasil melahirkan banyak lulusan yang sangat baik selama lebih dari dua dekade dan telah membangun reputasi baik di berbagai bidang. Namun, seperti institusi pendidikan lain, Jaya Jaya Institut menghadapi masalah besar yang terkait dengan sejumlah siswa yang tidak menyelesaikan program studinya atau yang dikenal sebagai dropout. Masalah ini sangat penting bagi pertumbuhan dan perkembangan institusi, karena rekor dropouts mempengaruhi citra institusi, kelulusan siswa, dan akhirnya prestasi dalam menggaet siswa baru untuk tahun-tahun berikutnya. Itu juga bisa menjadi tanda bahwa proses penerimaan mahasiswa, pembelajaran, dan dukungan akademik tidak memadai atau tidak layak dipertanyakan.
 
### Permasalahan Bisnis
Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.
 
Beberapa permasalahan yang perlu diidentifikasi dan diselesaikan adalah:
1. Bagaimana cara mendeteksi siswa yang berisiko tinggi untuk keluar dari sekolah sebelum waktu yang ditentukan?
2. Faktor-faktor apa saja yang mempengaruhi keputusan siswa untuk berhenti sekolah?
3. Apa langkah-langkah yang bisa diambil untuk meningkatkan retensi siswa dan memastikan lebih banyak siswa dapat menyelesaikan pendidikan mereka?

### Cakupan Proyek
1. Analisis Data: Menggunakan data yang tersedia untuk mengidentifikasi faktor-faktor utama yang berkontribusi pada kejadian dropout.
2. Visualisasi & Pelaporan: Membangun dashboard yang memungkinkan pemantauan dan analisis faktor-faktor tersebut dengan cara yang mudah dipahami.
3. Rekomendasi & Intervensi: Berdasarkan temuan analisis, memberikan saran untuk langkah-langkah yang bisa diambil untuk mengurangi tingkat dropout.

 
### Persiapan
Sumber Data: [Link Sumber Data]
 
Setup Lingkungan:
a. Menyiapkan Environment Conda:
Buat Environment Conda Baru:
1. Jalankan perintah untuk membuat environment baru dengan Python versi 3.12:
```
conda create --name myenv python=3.12
```
 
2. Aktifkan Environment Conda:
 
Untuk pengguna Windows:
```
conda activate myenv
```
 
3. Install Requirements:
Install semua dependensi yang terdaftar dalam file requirements.txt:
```
pip install -r requirements.txt
```
 
b. Menyiapkan Metabase:
1. Pull Image Docker Metabase:
Untuk mengunduh versi terbaru dari Metabase menggunakan Docker, jalankan perintah:
```
docker pull metabase/metabase:v0.46.4
```
 
2. Jalankan Metabase dengan Docker:
Untuk menjalankan Metabase di port 3000, gunakan perintah:
```
docker run -p 3000:3000 --name metabase metabase/metabase
```
 
3. Akses Metabase:
Buka browser dan akses Metabase pada alamat berikut:
```
http://localhost:3000/setup
```
 
4. Ikuti petunjuk untuk menyelesaikan proses setup Metabase.
 
 
c. Menyiapkan Database (Supabase):
1. Buat Akun dan Login ke Supabase:
- Daftarkan akun dan masuk ke dasbor Supabase melalui [tautan ini](https://supabase.com/dashboard/sign-in).
3. Buat Proyek Baru:
- Klik New Project untuk memulai proyek baru di Supabase.
5. Salin URI Database:
- Salin URI yang tersedia di pengaturan database Supabase Anda.
6. Kirim Dataset menggunakan SQLAlchemy:
Gunakan SQLAlchemy untuk mengirim dataset ke database dengan kode berikut:
```
from sqlalchemy import create_engine
URL = "postgresql://postgres.skwqbvmccluavdxflixa:kyq4kgX8cA8eH7Xy@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
engine = create_engine(URL)
df.to_sql('employee', engine)
```
 
Dengan langkah-langkah ini, Anda dapat menyiapkan environment, Metabase, dan Supabase untuk mulai bekerja dengan data dan membuat dashboard interaktif.
 
## Business Dashboard
Dashboard ini dirancang untuk memberikan wawasan yang lebih jelas kepada institut mengenai faktor-faktor yang mempengaruhi keputusan siswa dalam status mereka, seperti dropout, terdaftar, atau lulus. Dengan fitur visualisasi yang interaktif, dashboard ini memungkinkan institut untuk:

1. Memantau Jumlah Siswa Berdasarkan Status Mereka:
Melalui visualisasi data, dashboard ini memungkinkan tim untuk memantau jumlah siswa yang mengalami dropout, yang terdaftar, dan yang lulus. Ini memberikan gambaran yang jelas mengenai status siswa secara keseluruhan dalam institusi.

2. Menganalisis Distribusi Dropout Berdasarkan Faktor-Faktor Penting:
Dashboard ini juga memungkinkan analisis lebih lanjut mengenai faktor-faktor yang mempengaruhi keputusan siswa untuk dropout, seperti biaya pendidikan, status beasiswa, usia saat pendaftaran, status pernikahan, dan lain-lain. Ini memberikan wawasan mengenai kelompok siswa mana yang lebih rentan untuk dropout dan faktor-faktor sosial atau ekonomi apa yang mempengaruhi keputusan mereka.

3. Melihat Persentase Siswa dalam Setiap Kategori:
Melalui visualisasi yang menunjukkan persentase siswa yang termasuk dalam kategori dropout, terdaftar, dan lulus, tim dapat dengan mudah mengidentifikasi tren dan pola dalam status siswa. Ini juga memungkinkan perbandingan antar kategori secara langsung.

4. Identifikasi Faktor-Faktor yang Memengaruhi Dropout:
Dengan grafik yang menghubungkan berbagai faktor, seperti pemegang beasiswa atau status keuangan, tim dapat lebih mudah mengidentifikasi faktor yang paling berpengaruh terhadap keputusan siswa untuk keluar dari institusi. Ini memberikan informasi yang berguna untuk mengambil langkah-langkah intervensi yang lebih tepat sasaran.

![Business_Dashboard_1](https://github.com/Andriansyah2501/Data_Sciene_Edutech/blob/main/Ian2501_Dashboard_Metabase.jpg?raw=true)

 
## Menjalankan Sistem Machine Learning
Pada proyek ini, sebuah prototype telah disediakan untuk melakukan prediksi berdasarkan model yang telah dikembangkan. Untuk menjalankan prototype secara lokal, gunakan perintah berikut di terminal: streamlit run app.py.
dapat diakses secara publik [link streamlit sistem machine learning](https://appsds2andriansyah.streamlit.app/)

 
 
## Conclusion
Berdasarkan analisis data yang ditampilkan pada dashboard terkait status siswa, berikut adalah kesimpulan yang dapat diambil:

1. Usia: Siswa yang mengalami dropout mayoritas berada dalam rentang usia 15 hingga 22.5 tahun, yang menunjukkan bahwa siswa yang lebih muda, khususnya yang baru memasuki dunia pendidikan tinggi, memiliki tingkat dropout yang lebih tinggi.

2. Biaya Pendidikan: Berdasarkan grafik "Tuition Fees by Dropout", siswa yang membayar biaya pendidikan (dengan status "Yes" pada biaya) lebih cenderung untuk mengalami dropout. Hal ini mungkin mengindikasikan bahwa masalah finansial berkontribusi terhadap keputusan untuk berhenti.

3. Penerima Beasiswa: Siswa yang tidak menerima beasiswa (status "No" pada Scholarship Holder) cenderung memiliki tingkat dropout yang lebih tinggi, seperti yang terlihat pada grafik "Scholarship Holder by Dropout". Ini menunjukkan bahwa dukungan keuangan melalui beasiswa dapat menjadi faktor pendorong untuk menyelesaikan pendidikan.

4. Status Pernikahan: Grafik "Marital Status" menunjukkan bahwa siswa yang belum menikah (single) lebih sering mengalami dropout dibandingkan dengan mereka yang sudah menikah. Hal ini mungkin terkait dengan tanggung jawab atau stabilitas yang berbeda antara siswa dengan status pernikahan yang berbeda.

5. Status Keuangan (Debitur): Siswa yang memiliki status debitur (grafik "Debtor by Dropout") lebih cenderung untuk mengalami dropout. Ini menunjukkan bahwa masalah keuangan atau keterlambatan pembayaran dapat berkontribusi pada keputusan siswa untuk keluar.

Dengan menggunakan dashboard ini, tim institusi dapat mengambil langkah-langkah strategis berdasarkan analisis yang lebih mendalam terhadap faktor-faktor yang mempengaruhi tingkat dropout siswa. Berdasarkan informasi ini, institusi dapat:
1. Fokus pada kelompok usia tertentu atau siswa yang terhambat secara finansial untuk mengurangi tingkat dropout.
2. Merancang program beasiswa atau dukungan finansial yang lebih luas bagi siswa yang membutuhkan.
3. Menyusun kebijakan atau program yang mendukung siswa yang belum menikah atau yang memiliki masalah keuangan untuk meningkatkan tingkat retensi.
 
### Rekomendasi Action Items
 
1. Penerapan Sistem Pemantauan Siswa Berdasarkan Data:
Mengimplementasikan model prediktif yang telah dikembangkan untuk memantau siswa secara berkala. Sistem ini dapat digunakan oleh institusi untuk mengidentifikasi siswa yang berisiko tinggi mengalami dropout dan memberikan intervensi sejak awal, seperti bimbingan akademik atau bentuk dukungan lainnya.

2. Penyesuaian Kurikulum dan Program Studi:
Melakukan evaluasi terhadap program studi dengan tingkat dropout yang tinggi dan melakukan perbaikan pada kurikulum atau metode pengajaran. Salah satu pendekatan yang bisa dilakukan adalah dengan memberikan fleksibilitas lebih pada jadwal kursus atau menyediakan materi pendukung tambahan untuk membantu meringankan beban akademik bagi siswa.

3. Penguatan Program Dukungan Akademik dan Psikologis:
Berdasarkan temuan faktor-faktor risiko, institusi perlu memperkuat program dukungan akademik dan psikologis. Hal ini bisa meliputi peningkatan akses untuk bimbingan belajar, penyediaan sesi konseling, serta dukungan kesehatan mental bagi siswa yang teridentifikasi rentan.

