import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich import box
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from preprocessing import preprocess_text

# KONFIGURASI

console = Console()

DATASET_PATH = os.path.join(
    "dataset",
    "PRDECT-ID Dataset.csv"
)

OUTPUT_DIR = "output"

TEST_SIZE = 0.2
RANDOM_STATE = 42
ALPHA = 1.0

LABELS = ["negative", "positive"]

# FUNGSI TAMPILAN

def show_header():
    """Menampilkan judul program."""

    console.clear()

    console.print(
        Panel(
            "[bold cyan]ANALISIS SENTIMEN ULASAN TOKOPEDIA[/bold cyan]\n"
            "[white]Text Mining | TF-IDF | Multinomial Naive Bayes[/white]\n\n"
            "[dim]Program Analisis Data Mining[/dim]",
            border_style="cyan",
            box=box.DOUBLE,
            padding=(1, 3),
            expand=True
        )
    )


def show_section(title):
    """Menampilkan judul setiap bagian."""

    console.print()
    console.rule(
        f"[bold cyan]{title}[/bold cyan]",
        style="cyan"
    )


def show_status(message, status="SUCCESS"):
    """Menampilkan status proses."""

    if status == "SUCCESS":
        color = "green"
        symbol = "✔"
    elif status == "WARNING":
        color = "yellow"
        symbol = "!"
    else:
        color = "red"
        symbol = "✘"

    console.print(
        f"[{color}]{symbol} {message}[/{color}]"
    )


def show_metric_table(metrics):
    """Menampilkan hasil evaluasi dalam tabel."""

    table = Table(
        title="PERFORMA MODEL",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True
    )

    table.add_column(
        "Metrik",
        style="white",
        width=20
    )

    table.add_column(
        "Nilai",
        justify="center",
        style="bold green"
    )

    table.add_column(
        "Persentase",
        justify="right",
        style="cyan"
    )

    for name, value in metrics.items():
        table.add_row(
            name,
            f"{value:.4f}",
            f"{value * 100:.2f}%"
        )

    console.print(table)

# 1. MEMUAT DATASET

def load_dataset():

    show_section("1. PEMUATAN DATASET")

    if not os.path.exists(DATASET_PATH):
        console.print(
            f"[bold red]Dataset tidak ditemukan:[/bold red] "
            f"{DATASET_PATH}"
        )
        raise FileNotFoundError(DATASET_PATH)

    df = pd.read_csv(
        DATASET_PATH,
        sep=None,
        engine="python",
        encoding="utf-8-sig"
    )

    df.columns = df.columns.str.strip()

    required_columns = [
        "Customer Review",
        "Sentiment"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Kolom '{column}' tidak ditemukan. "
                f"Kolom tersedia: {df.columns.tolist()}"
            )

    data = df[required_columns].copy()

    info_table = Table(
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan"
    )

    info_table.add_column("Informasi")
    info_table.add_column("Nilai", justify="right")

    info_table.add_row(
        "Jumlah baris awal",
        f"{len(df):,}"
    )

    info_table.add_row(
        "Jumlah kolom awal",
        str(len(df.columns))
    )

    info_table.add_row(
        "Kolom ulasan",
        "Customer Review"
    )

    info_table.add_row(
        "Kolom target",
        "Sentiment"
    )

    console.print(info_table)

    return data

# 2. PERSIAPAN DATA

def prepare_data(data):

    show_section("2. PERSIAPAN DATA")

    initial_count = len(data)

    # Menghapus data kosong
    data = data.dropna(
        subset=["Customer Review", "Sentiment"]
    ).copy()

    # Menyeragamkan format
    data["Customer Review"] = (
        data["Customer Review"]
        .astype(str)
        .str.strip()
    )

    data["Sentiment"] = (
        data["Sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Menghapus ulasan kosong
    data = data[
        data["Customer Review"] != ""
    ]

    # Mempertahankan dua kelas sentimen
    data = data[
        data["Sentiment"].isin(LABELS)
    ]

    # Menghapus pasangan ulasan dan label duplikat
    data = data.drop_duplicates(
        subset=["Customer Review", "Sentiment"]
    )

    data = data.reset_index(drop=True)

    if data.empty:
        raise ValueError(
            "Tidak ada data yang tersisa setelah pembersihan."
        )

    if data["Sentiment"].nunique() != 2:
        raise ValueError(
            "Dataset harus memiliki kelas positive dan negative."
        )

    final_count = len(data)
    removed_count = initial_count - final_count

    summary = Table(
        title="RINGKASAN PEMBERSIHAN DATA",
        box=box.ROUNDED,
        header_style="bold cyan"
    )

    summary.add_column("Keterangan")
    summary.add_column("Jumlah", justify="right")

    summary.add_row(
        "Data awal",
        f"{initial_count:,}"
    )

    summary.add_row(
        "Data dihapus",
        f"{removed_count:,}"
    )

    summary.add_row(
        "Data akhir",
        f"[bold green]{final_count:,}[/bold green]"
    )

    console.print(summary)

    distribution = Table(
        title="DISTRIBUSI SENTIMEN",
        box=box.ROUNDED,
        header_style="bold cyan"
    )

    distribution.add_column("Sentimen")
    distribution.add_column("Jumlah", justify="right")
    distribution.add_column("Persentase", justify="right")

    counts = data["Sentiment"].value_counts()

    for label in LABELS:
        count = int(counts.get(label, 0))
        percentage = count / len(data) * 100

        distribution.add_row(
            label.capitalize(),
            f"{count:,}",
            f"{percentage:.2f}%"
        )

    console.print(distribution)

    # Memastikan setiap kelas memiliki data yang cukup
    class_counts = data["Sentiment"].value_counts()

    if len(class_counts) != 2:
        raise ValueError(
        "Dataset harus memiliki dua kelas: "
        "positive dan negative."
    )

    if class_counts.min() < 2:
        raise ValueError(
        "Setiap kelas harus memiliki minimal "
        "2 data untuk pembagian training dan testing."
    )

    return data

# 3. PREPROCESSING TEKS

def process_text(data):

    show_section("3. PREPROCESSING TEKS")

    console.print(
        "[dim]Tahapan: Case Folding → Cleansing → "
        "Tokenizing → Stopword Removal → Stemming[/dim]"
    )

    processed_reviews = []

    for review in track(
        data["Customer Review"],
        description="Memproses ulasan..."
    ):
        processed_reviews.append(
            preprocess_text(review)
        )

    data = data.copy()
    data["Cleaned Review"] = processed_reviews

    # Menghapus hasil preprocessing kosong
    data = data[
        data["Cleaned Review"].str.strip() != ""
    ].reset_index(drop=True)

    if data.empty:
        raise ValueError(
            "Semua ulasan kosong setelah preprocessing."
        )

    comparison = Table(
        title="PERBANDINGAN TEKS SEBELUM DAN SESUDAH",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True
    )

    comparison.add_column(
        "Teks Asli",
        ratio=1
    )

    comparison.add_column(
        "Hasil Preprocessing",
        ratio=1
    )

    for _, row in data.head(5).iterrows():
        comparison.add_row(
            row["Customer Review"][:180],
            row["Cleaned Review"][:180]
        )

    console.print(comparison)

    show_status(
        f"Preprocessing selesai. {len(data):,} ulasan tersisa."
    )

    return data

# 5. PEMBAGIAN DATA TRAINING DAN TESTING

def split_data(data):

    show_section("5. PEMBAGIAN DATA")

    X = data["Cleaned Review"]
    y = data["Sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    table = Table(
        title="KOMPOSISI DATA",
        box=box.ROUNDED,
        header_style="bold cyan"
    )

    table.add_column("Jenis Data")
    table.add_column("Jumlah", justify="right")
    table.add_column("Proporsi", justify="right")

    table.add_row(
        "Training",
        f"{len(X_train):,}",
        f"{len(X_train) / len(data) * 100:.1f}%"
    )

    table.add_row(
        "Testing",
        f"{len(X_test):,}",
        f"{len(X_test) / len(data) * 100:.1f}%"
    )

    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{len(data):,}[/bold]",
        "100%"
    )

    console.print(table)

    return X_train, X_test, y_train, y_test

# 5. PEMBOBOTAN TF-IDF

def perform_tfidf(X_train, X_test):

    show_section("6. PEMBOBOTAN KATA TF-IDF")

    vectorizer = TfidfVectorizer(
        norm="l2",
        use_idf=True,
        smooth_idf=True
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    table = Table(
        title="INFORMASI HASIL TF-IDF",
        box=box.ROUNDED,
        header_style="bold cyan"
    )

    table.add_column("Informasi")
    table.add_column("Nilai", justify="right")

    table.add_row(
        "Jumlah vocabulary",
        f"{len(vectorizer.vocabulary_):,}"
    )

    table.add_row(
        "Matriks training",
        str(X_train_tfidf.shape)
    )

    table.add_row(
        "Matriks testing",
        str(X_test_tfidf.shape)
    )

    table.add_row(
        "Normalisasi",
        "L2"
    )

    table.add_row(
        "IDF smoothing",
        "Aktif"
    )

    console.print(table)

    show_status("Pembobotan TF-IDF berhasil.")

    return vectorizer, X_train_tfidf, X_test_tfidf

# 7. ANALISIS TAMBAHAN TEXT MINING

def show_text_mining_analysis(
    data,
    X_train,
    y_train,
    X_train_tfidf,
    vectorizer
):
    show_section("7. ANALISIS TEXT MINING")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # A. STATISTIK TEKS

    console.print(
        "\n[bold cyan]A. STATISTIK DATA TEKS[/bold cyan]"
    )

    # Menghitung panjang setiap ulasan berdasarkan jumlah kata
    word_lengths = data["Cleaned Review"].apply(
        lambda text: len(text.split())
    )

    total_documents = len(data)
    total_words = int(word_lengths.sum())
    average_words = word_lengths.mean()
    min_words = int(word_lengths.min())
    max_words = int(word_lengths.max())

    vocabulary_size = len(vectorizer.get_feature_names_out())

    statistics = Table(
        title="STATISTIK KORPUS ULASAN",
        box=box.ROUNDED,
        header_style="bold cyan"
    )

    statistics.add_column("Informasi")
    statistics.add_column("Hasil", justify="right")

    statistics.add_row(
        "Jumlah dokumen",
        f"{total_documents:,}"
    )

    statistics.add_row(
        "Total kata dalam korpus",
        f"{total_words:,}"
    )

    statistics.add_row(
        "Rata-rata kata per ulasan",
        f"{average_words:.2f}"
    )

    statistics.add_row(
        "Ulasan terpendek",
        f"{min_words} kata"
    )

    statistics.add_row(
        "Ulasan terpanjang",
        f"{max_words} kata"
    )

    statistics.add_row(
        "Jumlah vocabulary training",
        f"{vocabulary_size:,}"
    )

    console.print(statistics)

    # Simpan statistik
    statistics_df = pd.DataFrame({
        "Informasi": [
            "Jumlah dokumen",
            "Total kata",
            "Rata-rata kata per ulasan",
            "Ulasan terpendek",
            "Ulasan terpanjang",
            "Jumlah vocabulary training"
        ],
        "Nilai": [
            total_documents,
            total_words,
            round(average_words, 2),
            min_words,
            max_words,
            vocabulary_size
        ]
    })

    statistics_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "statistik_text_mining.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # B. FREKUENSI KATA

    console.print(
        "\n[bold cyan]B. FREKUENSI KATA TERBANYAK[/bold cyan]"
    )

    # Menghitung frekuensi kata dari seluruh teks
    all_words = []

    for text in data["Cleaned Review"]:
        all_words.extend(text.split())

    word_frequency = Counter(all_words)

    top_words = word_frequency.most_common(20)

    frequency_df = pd.DataFrame(
        top_words,
        columns=["Kata", "Frekuensi"]
    )

    frequency_table = Table(
        title="TOP 20 KATA PALING SERING MUNCUL",
        box=box.ROUNDED,
        header_style="bold cyan"
    )

    frequency_table.add_column("Peringkat", justify="center")
    frequency_table.add_column("Kata")
    frequency_table.add_column("Frekuensi", justify="right")

    for rank, (word, frequency) in enumerate(
        top_words,
        start=1
    ):
        frequency_table.add_row(
            str(rank),
            word,
            f"{frequency:,}"
        )

    console.print(frequency_table)

    frequency_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "frekuensi_kata.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # Visualisasi frekuensi kata
    if not frequency_df.empty:

        plot_df = frequency_df.sort_values(
            "Frekuensi",
            ascending=True
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        bars = ax.barh(
            plot_df["Kata"],
            plot_df["Frekuensi"],
            color="#4C9BE8"
        )

        ax.bar_label(
            bars,
            padding=4,
            fmt="%d"
        )

        ax.set_title(
            "20 Kata Paling Sering Muncul dalam Ulasan Tokopedia",
            fontsize=13,
            fontweight="bold"
        )

        ax.set_xlabel("Frekuensi Kemunculan")
        ax.set_ylabel("Kata")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                OUTPUT_DIR,
                "frekuensi_kata.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close(fig)

    # C. KATA DOMINAN BERDASARKAN TF-IDF

    console.print(
        "\n[bold cyan]C. KATA DOMINAN BERDASARKAN SENTIMEN[/bold cyan]"
    )

    feature_names = vectorizer.get_feature_names_out()

    sentiment_results = []

    sentiment_top_words = {}

    for sentiment in LABELS:

        # Mengambil indeks data training berdasarkan sentimen
        indices = [
            i for i, label in enumerate(y_train)
            if label == sentiment
        ]

        if not indices:
            continue

        # Menghitung rata-rata bobot TF-IDF per kelas
        class_matrix = X_train_tfidf[indices]

        mean_tfidf = np.asarray(
            class_matrix.mean(axis=0)
        ).ravel()

        # Mengambil 15 kata dengan bobot rata-rata tertinggi
        top_indices = np.argsort(
            mean_tfidf
        )[::-1][:15]

        top_class_words = []

        for index in top_indices:

            word = feature_names[index]
            score = float(mean_tfidf[index])

            top_class_words.append(
                (word, score)
            )

            sentiment_results.append({
                "Sentimen": sentiment,
                "Kata": word,
                "Rata-rata TF-IDF": score
            })

        sentiment_top_words[sentiment] = top_class_words

        # Menampilkan tabel per kelas
        class_table = Table(
            title=f"15 KATA DOMINAN - {sentiment.upper()}",
            box=box.ROUNDED,
            header_style="bold cyan"
        )

        class_table.add_column(
            "Peringkat",
            justify="center"
        )

        class_table.add_column("Kata")

        class_table.add_column(
            "Rata-rata TF-IDF",
            justify="right"
        )

        for rank, (word, score) in enumerate(
            top_class_words,
            start=1
        ):
            class_table.add_row(
                str(rank),
                word,
                f"{score:.6f}"
            )

        console.print(class_table)

    # Menyimpan hasil TF-IDF per sentimen
    sentiment_df = pd.DataFrame(
        sentiment_results
    )

    sentiment_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "kata_dominan_per_sentimen.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # D. VISUALISASI TF-IDF PER SENTIMEN

    if sentiment_top_words:

        number_of_classes = len(sentiment_top_words)

        fig, axes = plt.subplots(
            1,
            number_of_classes,
            figsize=(8 * number_of_classes, 7),
            squeeze=False
        )

        for ax, (sentiment, words) in zip(
            axes.ravel(),
            sentiment_top_words.items()
        ):

            words_sorted = sorted(
                words,
                key=lambda item: item[1]
            )

            names = [
                item[0] for item in words_sorted
            ]

            scores = [
                item[1] for item in words_sorted
            ]

            bars = ax.barh(
                names,
                scores,
                color=(
                    "#E57373"
                    if sentiment == "negative"
                    else "#66BB6A"
                )
            )

            ax.bar_label(
                bars,
                fmt="%.4f",
                padding=3,
                fontsize=8
            )

            ax.set_title(
                f"Sentimen {sentiment.capitalize()}",
                fontweight="bold"
            )

            ax.set_xlabel("Rata-rata Bobot TF-IDF")
            ax.set_ylabel("Kata")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(
            "15 Kata dengan Rata-rata Bobot TF-IDF Tertinggi per Sentimen",
            fontsize=14,
            fontweight="bold"
        )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                OUTPUT_DIR,
                "tfidf_per_sentimen.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close(fig)

    show_status(
        "Analisis tambahan Text Mining berhasil diselesaikan."
    )

# 8. FREKUENSI KATA PER SENTIMEN

def show_word_frequency_by_sentiment(data):

    show_section("8. FREKUENSI KATA PER SENTIMEN")

    sentiment_frequency = {}

    all_results = []

    for sentiment in LABELS:

        # Mengambil ulasan berdasarkan kelas sentimen
        reviews = data.loc[
            data["Sentiment"] == sentiment,
            "Cleaned Review"
        ]

        words = []

        for review in reviews:
            words.extend(review.split())

        counter = Counter(words)

        top_words = counter.most_common(15)

        sentiment_frequency[sentiment] = top_words

        # Tabel frekuensi kata
        table = Table(
            title=f"15 KATA TERBANYAK - {sentiment.upper()}",
            box=box.ROUNDED,
            header_style="bold cyan"
        )

        table.add_column("Peringkat", justify="center")
        table.add_column("Kata")
        table.add_column("Frekuensi", justify="right")

        for rank, (word, frequency) in enumerate(
            top_words,
            start=1
        ):
            table.add_row(
                str(rank),
                word,
                f"{frequency:,}"
            )

            all_results.append({
                "Sentimen": sentiment,
                "Kata": word,
                "Frekuensi": frequency
            })

        console.print(table)

    # Menyimpan hasil ke CSV
    frequency_df = pd.DataFrame(all_results)

    frequency_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "frekuensi_kata_per_sentimen.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # Visualisasi dua kelas sentimen
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7)
    )

    for ax, sentiment in zip(
        axes,
        LABELS
    ):

        words = sentiment_frequency[sentiment]

        if not words:
            ax.set_title(
                f"Sentimen {sentiment.capitalize()} - Tidak ada data"
            )
            continue

        # Urutkan dari frekuensi terendah ke tertinggi
        words = sorted(
            words,
            key=lambda item: item[1]
        )

        labels = [item[0] for item in words]
        values = [item[1] for item in words]

        color = (
            "#E57373"
            if sentiment == "negative"
            else "#66BB6A"
        )

        bars = ax.barh(
            labels,
            values,
            color=color
        )

        ax.bar_label(
            bars,
            padding=3,
            fontsize=8
        )

        ax.set_title(
            f"Sentimen {sentiment.capitalize()}",
            fontweight="bold"
        )

        ax.set_xlabel("Frekuensi")
        ax.set_ylabel("Kata")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Frekuensi Kata Berdasarkan Sentimen",
        fontsize=14,
        fontweight="bold"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "frekuensi_kata_per_sentimen.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    show_status(
        "Analisis frekuensi kata per sentimen berhasil."
    )

# 13. ANALISIS KESALAHAN KLASIFIKASI

def show_misclassification_analysis(
    X_test,
    y_test,
    y_pred
):

    show_section("13. ANALISIS KESALAHAN KLASIFIKASI")

    results = pd.DataFrame({
        "Ulasan": X_test.values,
        "Sentimen Aktual": pd.Series(y_test).values,
        "Sentimen Prediksi": y_pred
    })

    # Menentukan status prediksi
    results["Status"] = (
        results["Sentimen Aktual"]
        == results["Sentimen Prediksi"]
    ).map({
        True: "Benar",
        False: "Salah"
    })

    # Mengambil prediksi yang salah
    errors = results[
        results["Status"] == "Salah"
    ].copy()

    # Mengelompokkan jenis kesalahan
    false_positive = errors[
        (errors["Sentimen Aktual"] == "negative")
        & (errors["Sentimen Prediksi"] == "positive")
    ]

    false_negative = errors[
        (errors["Sentimen Aktual"] == "positive")
        & (errors["Sentimen Prediksi"] == "negative")
    ]

    # Ringkasan kesalahan
    total = len(results)
    correct = total - len(errors)

    table = Table(
        title="RINGKASAN HASIL KLASIFIKASI",
        box=box.ROUNDED,
        header_style="bold cyan"
    )

    table.add_column("Keterangan")
    table.add_column("Jumlah", justify="right")
    table.add_column("Persentase", justify="right")

    table.add_row(
        "Prediksi benar",
        f"{correct:,}",
        f"{correct / total * 100:.2f}%" if total else "0.00%"
    )

    table.add_row(
        "Prediksi salah",
        f"{len(errors):,}",
        f"{len(errors) / total * 100:.2f}%" if total else "0.00%"
    )

    table.add_row(
        "False Positive",
        f"{len(false_positive):,}",
        f"{len(false_positive) / total * 100:.2f}%" if total else "0.00%"
    )

    table.add_row(
        "False Negative",
        f"{len(false_negative):,}",
        f"{len(false_negative) / total * 100:.2f}%" if total else "0.00%"
    )

    console.print(table)

    # Menampilkan contoh kesalahan klasifikasi
    error_table = Table(
        title="CONTOH ULASAN YANG SALAH DIKLASIFIKASIKAN",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True
    )

    error_table.add_column("No.", justify="center")
    error_table.add_column("Ulasan", max_width=55)
    error_table.add_column("Aktual", justify="center")
    error_table.add_column("Prediksi", justify="center")

    for index, (_, row) in enumerate(
        errors.head(15).iterrows(),
        start=1
    ):

        error_table.add_row(
            str(index),
            row["Ulasan"][:120],
            row["Sentimen Aktual"].capitalize(),
            row["Sentimen Prediksi"].capitalize()
        )

    if errors.empty:
        console.print(
            "[green]Tidak ditemukan kesalahan klasifikasi.[/green]"
        )
    else:
        console.print(error_table)

    # Menyimpan seluruh kesalahan klasifikasi
    errors.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "kesalahan_klasifikasi.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # Menyimpan ringkasan kesalahan
    error_summary = pd.DataFrame({
        "Jenis": [
            "Prediksi benar",
            "Prediksi salah",
            "False Positive",
            "False Negative"
        ],
        "Jumlah": [
            correct,
            len(errors),
            len(false_positive),
            len(false_negative)
        ]
    })

    error_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ringkasan_kesalahan.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    show_status(
        "Analisis kesalahan klasifikasi berhasil."
    )

    return errors

# 4. PERBANDINGAN SEBELUM DAN SESUDAH PREPROCESSING

def show_preprocessing_comparison(
    original_data,
    processed_data
):

    show_section("4. PERBANDINGAN PREPROCESSING")

    # Memastikan folder output tersedia sebelum menyimpan hasil.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Menghitung statistik teks
    def calculate_statistics(series):

        word_lists = [
            str(text).lower().split()
            for text in series
        ]

        total_words = sum(
            len(words)
            for words in word_lists
        )

        unique_words = len({
            word
            for words in word_lists
            for word in words
        })

        total_documents = len(series)

        average_length = (
            total_words / total_documents
            if total_documents else 0
        )

        return {
            "Jumlah dokumen": total_documents,
            "Total kata": total_words,
            "Kata unik": unique_words,
            "Rata-rata kata per ulasan": round(
                average_length, 2
            )
        }

    # Statistik sebelum preprocessing
    before = calculate_statistics(
        original_data["Customer Review"]
    )

    # Statistik sesudah preprocessing
    after = calculate_statistics(
        processed_data["Cleaned Review"]
    )

    # Tabel perbandingan
    table = Table(
        title="PERBANDINGAN SEBELUM DAN SESUDAH PREPROCESSING",
        box=box.ROUNDED,
        header_style="bold cyan"
    )

    table.add_column("Indikator")
    table.add_column("Sebelum", justify="right")
    table.add_column("Sesudah", justify="right")
    table.add_column("Perubahan", justify="right")

    for key in before:

        before_value = before[key]
        after_value = after[key]

        difference = after_value - before_value

        if key == "Rata-rata kata per ulasan":
            difference_text = f"{difference:+.2f}"
        else:
            difference_text = f"{difference:+,}"

        table.add_row(
            key,
            f"{before_value:,}",
            f"{after_value:,}",
            difference_text
        )

    console.print(table)

    # Menyimpan hasil perbandingan
    comparison_df = pd.DataFrame([
        {
            "Indikator": key,
            "Sebelum Preprocessing": before[key],
            "Sesudah Preprocessing": after[key],
            "Perubahan": after[key] - before[key]
        }
        for key in before
    ])

    comparison_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "perbandingan_preprocessing.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # Visualisasi perbandingan jumlah kata dan kata unik
    labels = [
        "Total Kata",
        "Kata Unik"
    ]

    before_values = [
        before["Total kata"],
        before["Kata unik"]
    ]

    after_values = [
        after["Total kata"],
        after["Kata unik"]
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_before = ax.bar(
        x - width / 2,
        before_values,
        width,
        label="Sebelum",
        color="#90A4AE"
    )

    bars_after = ax.bar(
        x + width / 2,
        after_values,
        width,
        label="Sesudah",
        color="#42A5F5"
    )

    ax.bar_label(
        bars_before,
        padding=3,
        fmt="%.0f"
    )

    ax.bar_label(
        bars_after,
        padding=3,
        fmt="%.0f"
    )

    ax.set_title(
        "Perbandingan Data Sebelum dan Sesudah Preprocessing",
        fontweight="bold"
    )

    ax.set_ylabel("Jumlah")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "perbandingan_preprocessing.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    show_status(
        "Perbandingan preprocessing berhasil diselesaikan."
    )

# 9. PELATIHAN MODEL

def train_model(X_train_tfidf, y_train):

    show_section("9. PEMODELAN MULTINOMIAL NAIVE BAYES")

    model = MultinomialNB(
        alpha=ALPHA
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    table = Table(
        title="KONFIGURASI MODEL",
        box=box.ROUNDED,
        header_style="bold cyan"
    )

    table.add_column("Parameter")
    table.add_column("Nilai")

    table.add_row(
        "Algoritma",
        "Multinomial Naive Bayes"
    )

    table.add_row(
        "Alpha",
        str(ALPHA)
    )

    table.add_row(
        "Jumlah data training",
        f"{len(y_train):,}"
    )

    table.add_row(
        "Kelas sentimen",
        "Negative, Positive"
    )

    console.print(table)

    show_status("Pelatihan model selesai.")

    return model

# 10. EVALUASI MODEL

def evaluate_model(model, X_test_tfidf, y_test):

    show_section("10. EVALUASI MODEL")

    # Melakukan prediksi terhadap data testing
    y_pred = model.predict(X_test_tfidf)

    # Memastikan label aktual dan prediksi sesuai
    y_test = pd.Series(y_test).reset_index(drop=True)
    y_pred = pd.Series(y_pred)

    # Menghitung metrik evaluasi
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test,
            y_pred,
            pos_label="positive",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            pos_label="positive",
            zero_division=0
        ),
        "F1-Score": f1_score(
            y_test,
            y_pred,
            pos_label="positive",
            zero_division=0
        )
    }

    # Menampilkan tabel performa model
    show_metric_table(metrics)

    # Membuat classification report.
    # Tidak menggunakan target_names agar nama kunci
    # tetap konsisten dengan LABELS (huruf kecil).
    report = classification_report(
        y_test,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0
    )

    report_table = Table(
        title="CLASSIFICATION REPORT",
        box=box.ROUNDED,
        header_style="bold cyan"
    )

    report_table.add_column("Kelas")
    report_table.add_column("Precision", justify="right")
    report_table.add_column("Recall", justify="right")
    report_table.add_column("F1-Score", justify="right")
    report_table.add_column("Support", justify="right")

    # Mengambil data berdasarkan label yang konsisten
    for label in LABELS:

        values = report[label]

        report_table.add_row(
            label.capitalize(),
            f"{values['precision']:.4f}",
            f"{values['recall']:.4f}",
            f"{values['f1-score']:.4f}",
            f"{int(values['support']):,}"
        )

    console.print(report_table)

    show_status("Evaluasi model berhasil diselesaikan.")

    return y_pred.to_numpy(), metrics

# 11. CONFUSION MATRIX

def show_confusion_matrix(y_test, y_pred):

    show_section("11. CONFUSION MATRIX")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=LABELS
    )

    table = Table(
        title="CONFUSION MATRIX",
        box=box.ROUNDED,
        header_style="bold cyan"
    )

    table.add_column("Aktual / Prediksi")
    table.add_column("Negative", justify="center")
    table.add_column("Positive", justify="center")

    table.add_row(
        "Negative",
        f"[green]{cm[0][0]:,}[/green]",
        f"[red]{cm[0][1]:,}[/red]"
    )

    table.add_row(
        "Positive",
        f"[red]{cm[1][0]:,}[/red]",
        f"[green]{cm[1][1]:,}[/green]"
    )

    console.print(table)

    console.print(
        "[dim]Baris menunjukkan label aktual, sedangkan "
        "kolom menunjukkan label prediksi.[/dim]"
    )

    # Visualisasi
    fig, ax = plt.subplots(figsize=(7, 5))

    image = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues"
    )

    ax.set_title("Confusion Matrix - Multinomial Naive Bayes")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])

    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=12
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig.savefig(
        os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    return cm

# 12. HASIL PREDIKSI DATA TESTING

def show_prediction_results(
    X_test,
    y_test,
    y_pred
):

    show_section("12. HASIL PREDIKSI DATA TESTING")

    results = pd.DataFrame({
        "Review": X_test.values,
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    results["Status"] = (
        results["Actual"] == results["Predicted"]
    ).map({
        True: "Benar",
        False: "Salah"
    })

    table = Table(
        title="15 CONTOH HASIL PREDIKSI",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True
    )

    table.add_column("No.", justify="center")
    table.add_column("Ulasan", max_width=45)
    table.add_column("Aktual", justify="center")
    table.add_column("Prediksi", justify="center")
    table.add_column("Status", justify="center")

    for index, row in results.head(15).iterrows():

        status_color = (
            "green" if row["Status"] == "Benar"
            else "red"
        )

        table.add_row(
            str(index + 1),
            row["Review"][:100],
            row["Actual"].capitalize(),
            row["Predicted"].capitalize(),
            f"[{status_color}]{row['Status']}[/{status_color}]"
        )

    console.print(table)

    correct = int(
        (results["Status"] == "Benar").sum()
    )

    incorrect = int(
        (results["Status"] == "Salah").sum()
    )

    summary = Table(
        title="RINGKASAN PREDIKSI",
        box=box.ROUNDED
    )

    summary.add_column("Keterangan")
    summary.add_column("Jumlah", justify="right")

    summary.add_row("Prediksi benar", f"[green]{correct:,}[/green]")
    summary.add_row("Prediksi salah", f"[red]{incorrect:,}[/red]")
    summary.add_row("Total pengujian", f"{len(results):,}")

    console.print(summary)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results.to_csv(
        os.path.join(OUTPUT_DIR, "hasil_prediksi.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    show_status("Hasil prediksi disimpan ke output/hasil_prediksi.csv")

    return results

# 14. VISUALISASI DISTRIBUSI SENTIMEN

def show_sentiment_distribution(data):

    show_section("14. VISUALISASI DISTRIBUSI SENTIMEN")

    counts = data["Sentiment"].value_counts().reindex(
        LABELS,
        fill_value=0
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        ["Negative", "Positive"],
        counts.values,
        color=["#E57373", "#66BB6A"],
        width=0.55
    )

    ax.set_title("Distribusi Sentimen Ulasan Tokopedia")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Jumlah Ulasan")

    ax.bar_label(
        bars,
        labels=[f"{value:,}" for value in counts.values],
        padding=4
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "distribusi_sentimen.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    show_status(
        "Visualisasi distribusi sentimen berhasil disimpan."
    )

# PROGRAM UTAMA

def main():

    try:
        # 1. MEMUAT DATASET
        show_header()

        data = load_dataset()

        # 2. PERSIAPAN DATA
        data = prepare_data(data)

        # Menyimpan salinan data sebelum preprocessing
        original_data = data.copy()

        # 3. PREPROCESSING TEKS
        data = process_text(data)

        # 4. PERBANDINGAN SEBELUM DAN SESUDAH PREPROCESSING
        show_preprocessing_comparison(
            original_data,
            data
        )

        # 5. PEMBAGIAN DATA TRAINING DAN TESTING
        X_train, X_test, y_train, y_test = split_data(data)

        # 6. PEMBOBOTAN KATA MENGGUNAKAN TF-IDF
        vectorizer, X_train_tfidf, X_test_tfidf = perform_tfidf(
            X_train,
            X_test
        )

        # 7. ANALISIS TAMBAHAN TEXT MINING
        show_text_mining_analysis(
            data,
            X_train,
            y_train,
            X_train_tfidf,
            vectorizer
        )

        # 8. FREKUENSI KATA BERDASARKAN SENTIMEN
        show_word_frequency_by_sentiment(data)

        # 9. PELATIHAN MODEL MULTINOMIAL NAIVE BAYES
        model = train_model(
            X_train_tfidf,
            y_train
        )

        # 10. EVALUASI MODEL
        y_pred, metrics = evaluate_model(
            model,
            X_test_tfidf,
            y_test
        )

        # 11. CONFUSION MATRIX
        show_confusion_matrix(
            y_test,
            y_pred
        )

        # 12. HASIL PREDIKSI DATA TESTING
        show_prediction_results(
            X_test,
            y_test,
            y_pred
        )

        # 13. ANALISIS KESALAHAN KLASIFIKASI
        show_misclassification_analysis(
            X_test,
            y_test,
            y_pred
        )

        # 14. VISUALISASI DISTRIBUSI SENTIMEN
        show_sentiment_distribution(data)

        # PENUTUP
        console.print()

        console.print(
            Panel(
                "[bold green]ANALISIS SENTIMEN BERHASIL DISELESAIKAN[/bold green]\n\n"
                "Seluruh proses Text Mining telah selesai.\n"
                "Hasil analisis tersedia pada folder [cyan]output/[/cyan].",
                border_style="green",
                padding=(1, 2)
            )
        )

    except Exception as error:

        console.print()

        console.print(
            Panel(
                f"[bold red]PROGRAM GAGAL DIJALANKAN[/bold red]\n\n"
                f"{str(error)}",
                border_style="red"
            )
        )

        sys.exit(1)


if __name__ == "__main__":
    main()