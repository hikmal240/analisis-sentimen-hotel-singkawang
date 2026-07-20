"""
pipeline.py
=============================================================================
Modul inti (importable) — Analisis Sentimen Hotel di Kota Singkawang
Hybrid Lexicon Labeling + Multinomial Naive Bayes

Direfactor dari lanjutan_analisis_hybrid_lexicon.py agar semua langkah
(labeling, split, TF-IDF, training, evaluasi) dibungkus jadi fungsi yang
bisa dipanggil dari app.py (Streamlit), notebook, atau script lain,
BUKAN kode top-level yang otomatis jalan saat modul di-import.

Fungsi utama yang dipakai dari luar:
    jalankan_pipeline(csv_files: dict) -> dict
=============================================================================
"""

import os
import re
import csv
import tempfile
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

RANDOM_STATE = 42

TFIDF_PARAMS = dict(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)


# =============================================================================
# 1. MEMUAT DATA HASIL PREPROCESSING (CSV)
# =============================================================================
def muat_csv_preprocessing(path: str, hotel_name: str) -> tuple:
    """
    Parser khusus untuk CSV 'No,Hotel,Komentar Asli,Hasil Preprocessing,Label'
    yang kolom 'Komentar Asli'-nya sering mengandung koma tanpa quoting yang
    konsisten (CSV reader standar sering salah bagi kolom di kasus ini).
    """
    # Pastikan path absolut berbasis direktori script jika diberikan path relatif
    if not os.path.isabs(path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, path)

    raw = open(path, encoding='utf-8-sig').read().replace('\r\n', '\n')
    body = raw.split('\n', 1)[1]  # buang header
    record_start = re.compile(r'(?=^\d+,' + re.escape(hotel_name) + r',)', re.MULTILINE)
    parts = [p.strip('\n') for p in record_start.split(body) if p.strip()]
    row_re = re.compile(
        r'^(\d+),' + re.escape(hotel_name) + r',(.*),([^,]*),(Positif|Netral|Negatif)\s*$',
        re.DOTALL
    )
    rows, gagal = [], 0
    for p in parts:
        m = row_re.match(p)
        if not m:
            gagal += 1
            continue
        no, asli, prep, label = m.groups()
        rows.append({
            'no': no.strip(),
            'komentar_asli': asli.strip(),
            'komentar': prep.strip(),
            'label_lama': label.strip(),
        })
    df = pd.DataFrame(rows)
    return df, gagal


# =============================================================================
# 2. LEKSIKON HYBRID (InSet + filter + domain + kontras + intensifier)
# =============================================================================
ABAIKAN = {
    'ya', 'yah', 'abang', 'maksud', 'berapa', 'juta', 'penginapan', 'biar', 'banyak',
    'murah', 'ada', 'itu', 'ini', 'saja', 'sih', 'deh', 'nih', 'kan', 'lah', 'tuh',
    'kok', 'dong', 'aja', 'pun', 'juga', 'lagi', 'masih', 'sudah', 'akan', 'jadi',
    'bisa', 'bs', 'dlu', 'dulu', 'besok', 'kesitu', 'sana', 'sini', 'situ',
    'aamiin', 'amin', 'insya', 'allah', 'alhamdulillah', 'astaghfirullah',
    'kolam', 'harga', 'info', 'liburan', 'malam', 'musim', 'kamar', 'hotel',
    'resort', 'villa', 'pantai', 'tempat', 'wahana', 'fasilitas',
}
DOMAIN_POSITIF = {
    'bagus': 3, 'keren': 3, 'mantap': 3, 'puas': 3, 'nyaman': 3, 'bersih': 3, 'indah': 2,
    'sejuk': 2, 'luas': 2, 'mewah': 3, 'rapi': 2, 'ramah': 3, 'terbaik': 4, 'rekomendasi': 3,
    'istimewa': 3, 'seru': 2, 'menyenangkan': 3, 'berkesan': 2, 'lengkap': 2, 'terawat': 2,
    'strategis': 2, 'terjangkau': 2, 'hemat': 2, 'sempurna': 3, 'worth': 2, 'recommended': 3,
    'amazing': 4, 'gercep': 2, 'responsif': 2, 'estetik': 2, 'aesthetic': 2, 'kece': 2,
    'sultan': 2, 'juara': 3,
}
DOMAIN_NEGATIF = {
    'kotor': -3, 'jorok': -3, 'bau': -2, 'lembab': -2, 'kumuh': -3, 'rusak': -3, 'bocor': -2,
    'sempit': -2, 'pengap': -2, 'jelek': -3, 'buruk': -3, 'payah': -2, 'mengecewakan': -3,
    'lambat': -2, 'lamban': -2, 'cuek': -2, 'jutek': -2, 'kasar': -3, 'mahal': -2, 'kecewa': -3,
    'trauma': -3, 'menyesal': -3, 'marah': -2, 'kesal': -2, 'berisik': -2, 'bising': -2,
    'kapok': -3, 'horor': -3, 'nyamuk': -2, 'serem': -2, 'seram': -2, 'takut': -2,
    'nyesel': -3, 'parah': -2, 'zonk': -2,
}
KATA_NEGASI = {'tidak', 'bukan', 'kurang', 'belum', 'tak', 'tanpa', 'jangan', 'sulit', 'susah', 'mustahil', 'gagal'}
KATA_TANYA = {'apa', 'siapa', 'kapan', 'dimana', 'bagaimana', 'mengapa', 'kenapa', 'berapa', 'gimana', 'knp', 'knpa'}
KATA_KONTRAS = {'tapi', 'tetapi', 'namun', 'sayangnya', 'cuma', 'hanya'}
INTENSIFIER = {'banget': 1.5, 'bgt': 1.5, 'sangat': 1.5, 'sekali': 1.4, 'terlalu': 1.3}


def unduh_inset() -> tuple:
    """Unduh InSet Lexicon asli (fajri91/InSet) dari GitHub. Fallback ke dict kosong bila gagal."""
    base = 'https://raw.githubusercontent.com/fajri91/InSet/master/'
    p_pos = os.path.join(tempfile.gettempdir(), 'inset_pos.tsv')
    p_neg = os.path.join(tempfile.gettempdir(), 'inset_neg.tsv')
    try:
        urllib.request.urlretrieve(base + 'positive.tsv', p_pos)
        urllib.request.urlretrieve(base + 'negative.tsv', p_neg)
    except Exception:
        return {}, {}
    pos, neg = {}, {}
    with open(p_pos, encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            if len(row) >= 2:
                try:
                    pos[row[0].strip().lower()] = int(row[1])
                except ValueError:
                    pass
    with open(p_neg, encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            if len(row) >= 2:
                try:
                    neg[row[0].strip().lower()] = int(row[1])
                except ValueError:
                    pass
    return pos, neg


def bangun_leksikon_hybrid(raw_pos: dict, raw_neg: dict) -> tuple:
    """Filter InSet mentah (buang ambigu & kata topikal-netral) + gabung leksikon domain."""
    core_pos = {k: v for k, v in raw_pos.items()
                if k not in raw_neg and k not in ABAIKAN and abs(v) >= 2 and len(k) >= 4}
    core_neg = {k: v for k, v in raw_neg.items()
                if k not in raw_pos and k not in ABAIKAN and abs(v) >= 2 and len(k) >= 4}
    kata_positif = {**core_pos, **DOMAIN_POSITIF}
    kata_negatif = {**core_neg, **DOMAIN_NEGATIF}
    for k in DOMAIN_POSITIF:
        kata_negatif.pop(k, None)
    for k in DOMAIN_NEGATIF:
        kata_positif.pop(k, None)
    return kata_positif, kata_negatif


def _score_clause(tokens, kata_positif, kata_negatif, weight=1.0):
    skor = 0.0
    negasi_aktif, window_counter, window_max = False, 0, 2
    pending_mult = 1.0
    for token in tokens:
        if token in INTENSIFIER:
            pending_mult = INTENSIFIER[token]
            continue
        if token in KATA_NEGASI:
            negasi_aktif, window_counter = True, 0
            continue
        bobot = kata_positif.get(token)
        if bobot is None:
            bobot = kata_negatif.get(token)
        if bobot is not None:
            bobot = bobot * pending_mult
            pending_mult = 1.0
            if negasi_aktif:
                skor += -bobot * weight
                negasi_aktif, window_counter = False, 0
            else:
                skor += bobot * weight
        else:
            pending_mult = 1.0
            if negasi_aktif:
                window_counter += 1
                if window_counter >= window_max:
                    negasi_aktif, window_counter = False, 0
    return skor


def labeling_hybrid(text: str, kata_positif: dict, kata_negatif: dict, min_evidence: float = 2.0) -> str:
    """Label sentimen: leksikon hybrid + negasi window + klausa kontras + intensifier."""
    text = str(text).lower()
    tokens_all = text.split()
    is_question = ('?' in text) or any(t in KATA_TANYA for t in tokens_all)

    kontras_idx = None
    for i, t in enumerate(tokens_all):
        if t in KATA_KONTRAS:
            kontras_idx = i
            break

    if kontras_idx is not None:
        before, after = tokens_all[:kontras_idx], tokens_all[kontras_idx + 1:]
        skor = (_score_clause(before, kata_positif, kata_negatif, weight=0.5) +
                _score_clause(after, kata_positif, kata_negatif, weight=1.5))
    else:
        skor = _score_clause(tokens_all, kata_positif, kata_negatif, weight=1.0)

    if is_question and abs(skor) < min_evidence * 2:
        return 'Netral'
    if skor >= min_evidence:
        return 'Positif'
    elif skor <= -min_evidence:
        return 'Negatif'
    return 'Netral'


# =============================================================================
# 3. MODELING: SPLIT + TF-IDF + MULTINOMIAL NAIVE BAYES
# =============================================================================
def latih_dan_evaluasi(df: pd.DataFrame) -> dict:
    """Split 80:20 stratified -> TF-IDF (fit hanya di train) -> Multinomial NB -> evaluasi."""
    
    # KONVERSI EKSPLISIT ke NumPy Array (mencegah error PyArrow pada Python 3.14/Streamlit Cloud)
    texts = np.array(df['komentar'].astype(str).tolist(), dtype=object)
    y = np.array(df['label'].astype(str).tolist(), dtype=object)

    unique_labels, counts = np.unique(y, return_counts=True)
    bisa_stratify = (len(unique_labels) >= 2) and (counts.min() >= 2)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, y, test_size=0.20, random_state=RANDOM_STATE,
        stratify=y if bisa_stratify else None
    )

    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)

    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    labels_cm = ['Positif', 'Negatif', 'Netral']
    cm = confusion_matrix(y_test, y_pred, labels=labels_cm)
    rep_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    rep_str = classification_report(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    return {
        'clf': clf, 'tfidf': tfidf,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test, 'y_pred': y_pred,
        'accuracy': acc, 'report': rep_dict, 'report_str': rep_str,
        'cm': cm, 'labels': labels_cm,
        'n_train': len(y_train), 'n_test': len(y_test),
    }


def top_fitur_tfidf(res: dict, n: int = 10):
    """Ambil n fitur TF-IDF teratas (berdasarkan total bobot di data latih)."""
    feature_names = np.array(res['tfidf'].get_feature_names_out())
    sums = np.asarray(res['X_train'].sum(axis=0)).flatten()
    top_idx = sums.argsort()[-n:][::-1]
    return list(zip(feature_names[top_idx], sums[top_idx]))


# =============================================================================
# 4. FUNGSI UTAMA — dipanggil dari app.py
# =============================================================================
def jalankan_pipeline(csv_files: dict, progress_cb=None) -> dict:
    """
    Jalankan seluruh pipeline: muat CSV -> label hybrid -> split -> TF-IDF ->
    Multinomial NB -> evaluasi, untuk setiap hotel di csv_files.
    """
    def log(msg):
        if progress_cb:
            progress_cb(msg)

    warnings_list = []

    log("Mengunduh InSet Lexicon dari GitHub...")
    raw_pos, raw_neg = unduh_inset()
    kata_positif, kata_negatif = bangun_leksikon_hybrid(raw_pos, raw_neg)
    log(f"Leksikon hybrid siap: {len(kata_positif)} kata positif, {len(kata_negatif)} kata negatif.")

    clean_data = {}
    for hotel_name, path in csv_files.items():
        log(f"Memuat & melabel: {hotel_name}...")
        df, gagal = muat_csv_preprocessing(path, hotel_name)
        if gagal:
            warnings_list.append(f"{hotel_name}: {gagal} baris gagal di-parse dan dilewati.")
        df['label'] = df['komentar'].apply(lambda t: labeling_hybrid(t, kata_positif, kata_negatif))
        clean_data[hotel_name] = df

    results = {}
    for hotel_name, df in clean_data.items():
        log(f"Melatih Multinomial Naive Bayes: {hotel_name}...")
        results[hotel_name] = latih_dan_evaluasi(df)

    return {
        'clean_data': clean_data,
        'results': results,
        'lexicon_size': (len(kata_positif), len(kata_negatif)),
        'warnings': warnings_list,
    }


def ringkasan_distribusi_total(clean_data: dict) -> dict:
    """Distribusi sentimen gabungan seluruh hotel."""
    total = Counter()
    for df in clean_data.values():
        total.update(df['label'])
    n = sum(total.values()) or 1
    return {
        'counts': dict(total),
        'pct': {k: v / n * 100 for k, v in total.items()},
        'n_total': n,
    }
