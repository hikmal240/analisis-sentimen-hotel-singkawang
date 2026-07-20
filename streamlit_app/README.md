# Analisis Sentimen Hotel di Kota Singkawang

Aplikasi Streamlit untuk analisis sentimen komentar TikTok terhadap 4 hotel
di Kota Singkawang, menggunakan **Hybrid Lexicon Labeling** (InSet Lexicon
yang difilter + leksikon domain hotel + penanganan negasi/kontras/intensifier)
dan **Multinomial Naïve Bayes** untuk klasifikasi.

Tugas Akhir — Hikmal, NIM 3202302015, D3 Manajemen Informatika,
Politeknik Negeri Sambas, 2026.

## Struktur Proyek

```
.
├── app.py              # Entry point Streamlit
├── pipeline.py          # Modul inti: labeling hybrid, TF-IDF, Naive Bayes
├── requirements.txt
├── data/                 # 4 CSV hasil preprocessing (No,Hotel,Komentar Asli,Hasil Preprocessing,Label)
│   ├── HOTEL_MAHKOTA_SINGKAWANG.csv
│   ├── HOTEL_SWISS-BELHOTEL_SINGKAWANG.csv
│   ├── HOTEL_DAYANG_RESORT_SINGKAWANG.csv
│   └── HOTEL_HORISON_ULTIMA_SINGKAWANG.csv
└── README.md
```

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

## Deploy ke Streamlit Community Cloud

1. Push repo ini ke GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io), login dengan akun GitHub.
3. Klik **New app** → pilih repo ini, branch `main`, main file `app.py`.
4. Klik **Deploy**.

## Fitur Aplikasi

- **Ringkasan Semua Hotel** — metrik agregat, distribusi sentimen gabungan, perbandingan akurasi antar hotel.
- **Detail per Hotel** — distribusi sentimen, perbandingan label lama vs baru, confusion matrix, top fitur TF-IDF, tabel komentar (bisa difilter & diunduh).
- **Coba Label Komentar Sendiri** — form interaktif untuk menguji fungsi `labeling_hybrid()` terhadap kalimat baru.

## Metodologi Singkat

| Tahap | Detail |
|---|---|
| Labeling | InSet Lexicon (difilter dari kata ambigu/topikal) + ~90 kata domain hotel, negasi window, klausa kontras (1.5x/0.5x), intensifier, ambang bukti minimum |
| Fitur | TF-IDF, `max_features=1000`, `ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True` |
| Model | Multinomial Naïve Bayes, `alpha=1.0` |
| Split | 80:20 stratified, `random_state=42` |
