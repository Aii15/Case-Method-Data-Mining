# Case-Method-Data-Mining
Repositori ini berisi kode program dari tugas case method mata kuliah Data Mining 2026. Dibuat oleh Muhammad Ayman dengan NIM L0124137.

Program ini melakukan analisis sentimen ulasan Tokopedia menggunakan preprocessing teks, TF-IDF, dan algoritma Multinomial Naive Bayes.

## Persyaratan

- Python 3
- `pip`
- Git (jika ingin mengunduh repositori melalui terminal)

## Langkah Menjalankan Program

### 1. Masuk ke direktori proyek

Gunakan perintah `cd` sesuai folder penyimpanan proyek.

### 2. Buat dan aktifkan virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Jika Windows menolak eksekusi skrip aktivasi, gunakan Command Prompt:

```bat
venv\Scripts\activate.bat
```

Atau buka PowerShell dengan izin sementara:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3. Perbarui pip dan instal dependensi

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Pastikan dataset tersedia

Pastikan file dataset berada pada lokasi berikut:

```text
dataset/PRDECT-ID Dataset.csv
```

Dataset wajib memiliki kolom `Customer Review` dan `Sentiment`.

### 5. Jalankan program

```powershell
python main.py
```

Program akan:

1. Membaca dataset ulasan.
2. Melakukan preprocessing teks.
3. Membagi data menjadi data training dan testing.
4. Melakukan pembobotan TF-IDF.
5. Melatih model Multinomial Naive Bayes.
6. Menampilkan metrik evaluasi dan confusion matrix.
7. Menyimpan hasil analisis ke folder `output/`.

## Struktur Program

```text
main.py                     # Alur utama analisis sentimen
preprocessing.py            # Fungsi preprocessing teks Bahasa Indonesia
requirements.txt            # Daftar pustaka Python
dataset/                    # Data ulasan
output/                     # Grafik dan file hasil analisis
```

## Menonaktifkan Virtual Environment

Setelah selesai menjalankan program, gunakan perintah berikut:

```powershell
deactivate
```
