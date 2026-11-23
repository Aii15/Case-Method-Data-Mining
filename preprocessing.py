
import re
import html

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# Inisialisasi stemmer Bahasa Indonesia
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Mengambil daftar stopword
stopword_factory = StopWordRemoverFactory()
stopwords = set(stopword_factory.get_stop_words())

# Kata negasi yang harus dipertahankan
negation_words = {
    "tidak",
    "bukan",
    "belum",
    "jangan",
    "kurang",
    "tanpa"
}

stopwords = stopwords - negation_words


def case_folding(text):
    """Mengubah teks menjadi huruf kecil."""
    return str(text).lower()


def cleansing(text):
    """Membersihkan teks dari URL, HTML, angka, dan simbol."""
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenizing(text):
    """Memecah teks menjadi token."""
    return text.split()


def stopword_removal(tokens):
    """Menghapus stopword tanpa menghilangkan kata negasi."""
    return [
        word for word in tokens
        if word not in stopwords
    ]


def stemming(tokens):
    """Mengubah kata ke bentuk dasarnya."""
    text = " ".join(tokens)
    result = stemmer.stem(text)

    return result.split()


def preprocess_text(text):
    """Menjalankan seluruh tahapan preprocessing."""
    text = case_folding(text)
    text = cleansing(text)

    tokens = tokenizing(text)
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)

    return " ".join(tokens)
