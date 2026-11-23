# Case-Method-Data-Mining
Repositori ini berisi kode program dari tugas case method mata kuliah Data Mining 2026. Dibuat oleh Muhammad Ayman dengan NIM L0124137.

Program ini melakukan analisis sentimen ulasan Tokopedia menggunakan preprocessing teks, TF-IDF, dan algoritma Multinomial Naive Bayes.

## **1. Persiapan**

Pastikan Python dan VS Code sudah terpasang.

Pastikan semua file dan dataset tersedia. File dataset harus berada pada lokasi berikut:

```text
dataset/PRDECT-ID Dataset.csv
```

Dataset harus memiliki kolom `Customer Review` dan `Sentiment`.

## **2. Instalasi library**

Buka terminal VS Code dan jalankan:

```bat
python -m venv venv
```

Aktifkan virtual environment di Windows:

```bat
venv\Scripts\activate
```

Instal library:

```bat
pip install -r requirements.txt
```

## **3. Struktur program**

```text
main.py
preprocessing.py
requirements.txt
dataset/
    PRDECT-ID Dataset.csv
output/
```

## **4. Menjalankan program**

Setelah semua file dan dataset tersedia, jalankan perintah berikut di terminal VS Code:

```bat
python main.py
```

Pastikan terminal sedang berada di folder utama project.

Setelah program selesai dijalankan, hasil analisis dan visualisasi akan tersimpan di folder `output/`.

## **5. Menonaktifkan virtual environment**

Untuk keluar dari virtual environment, jalankan:

```bat
deactivate
```
