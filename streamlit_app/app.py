"""
app.py
=============================================================================
Streamlit App — Analisis Sentimen Hotel di Kota Singkawang
Hybrid Lexicon Labeling + Multinomial Naive Bayes

Jalankan lokal   : streamlit run app.py
Deploy           : Streamlit Community Cloud (share.streamlit.io), main file app.py
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline import jalankan_pipeline, top_fitur_tfidf, ringkasan_distribusi_total

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="Analisis Sentimen Hotel Singkawang",
    page_icon="📊",
    layout="wide",
)

# Gunakan os.path untuk penentuan path absolut berbasis lokasi file app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILES = {
    "Hotel Mahkota Singkawang": os.path.join(BASE_DIR, "data", "HOTEL_MAHKOTA_SINGKAWANG.csv"),
    "Hotel Swiss-Belhotel Singkawang": os.path.join(BASE_DIR, "data", "HOTEL_SWISS-BELHOTEL_SINGKAWANG.csv"),
    "Hotel Dayang Resort Singkawang": os.path.join(BASE_DIR, "data", "HOTEL_DAYANG_RESORT_SINGKAWANG.csv"),
    "Hotel Horison Ultima Singkawang": os.path.join(BASE_DIR, "data", "HOTEL_HORISON_ULTIMA_SINGKAWANG.csv"),
}
WARNA = {"Positif": "#4CAF50", "Negatif": "#F44336", "Netral": "#2196F3"}

st.title("📊 Analisis Sentimen Hotel di Kota Singkawang")
st.caption(
    "Implementasi Multinomial Naïve Bayes untuk Analisis Sentimen Komentar "
    "TikTok — Hybrid Lexicon Labeling (revisi pasca sidang)"
)

# =============================================================================
# CACHE: jalankan pipeline sekali saja (bukan tiap interaksi UI)
# =============================================================================
@st.cache_data(show_spinner=False)
def _run_pipeline():
    log_box = st.empty()
    logs = []

    def progress_cb(msg):
        logs.append(msg)
        log_box.info("\n\n".join(logs[-5:]))

    hasil = jalankan_pipeline(CSV_FILES, progress_cb=progress_cb)
    log_box.empty()
    return hasil


with st.spinner("Menjalankan pipeline (unduh leksikon, labeling, training)..."):
    hasil = _run_pipeline()

clean_data = hasil["clean_data"]
results = hasil["results"]
pos_n, neg_n = hasil["lexicon_size"]

if hasil["warnings"]:
    with st.expander("⚠️ Peringatan saat memuat data"):
        for w in hasil["warnings"]:
            st.warning(w)

st.success(
    f"Pipeline selesai. Leksikon hybrid: {pos_n} kata positif, {neg_n} kata negatif."
)

# =============================================================================
# SIDEBAR — Navigasi & Info
# =============================================================================
st.sidebar.header("Navigasi")
halaman = st.sidebar.radio(
    "Pilih Tampilan",
    ["Ringkasan Semua Hotel", "Detail per Hotel", "Coba Label Komentar Sendiri"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Metodologi**\n\n"
    "- Leksikon: InSet (difilter) + domain hotel\n"
    "- Negasi, klausa kontras, intensifier\n"
    "- TF-IDF (max_features=1000, ngram (1,2))\n"
    "- Multinomial Naive Bayes (alpha=1.0)\n"
    "- Split 80:20 stratified, random_state=42"
)

# =============================================================================
# HALAMAN 1: RINGKASAN SEMUA HOTEL
# =============================================================================
if halaman == "Ringkasan Semua Hotel":
    dist_total = ringkasan_distribusi_total(clean_data)
    avg_acc = np.mean([r["accuracy"] for r in results.values()]) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Komentar", f"{dist_total['n_total']:,}")
    col2.metric("Rata-rata Accuracy", f"{avg_acc:.2f}%")
    col3.metric("Sentimen Dominan", max(dist_total["counts"], key=dist_total["counts"].get))
    col4.metric("vs Random Baseline", f"+{avg_acc - 33.3:.1f} poin")

    st.subheader("Rekap per Hotel")
    rows = []
    for hotel_name, res in results.items():
        df = clean_data[hotel_name]
        vc = df["label"].value_counts()
        rows.append({
            "Hotel": hotel_name,
            "Total Data": len(df),
            "Data Latih": res["n_train"],
            "Data Uji": res["n_test"],
            "Accuracy": f"{res['accuracy']*100:.2f}%",
            "Sentimen Dominan": vc.idxmax(),
            "Positif": int(vc.get("Positif", 0)),
            "Netral": int(vc.get("Netral", 0)),
            "Negatif": int(vc.get("Negatif", 0)),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Distribusi Sentimen Gabungan")
    c1, c2 = st.columns([1, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(5, 5))
        labels = list(dist_total["counts"].keys())
        vals = list(dist_total["counts"].values())
        ax.pie(vals, labels=[f"{l} ({v:,})" for l, v in zip(labels, vals)],
               autopct="%1.1f%%", colors=[WARNA[l] for l in labels], startangle=90)
        ax.set_title("Distribusi Sentimen Total (4 Hotel)")
        st.pyplot(fig)
    with c2:
        st.bar_chart(pd.Series(dist_total["counts"]))

    st.subheader("Perbandingan Akurasi per Hotel")
    acc_series = pd.Series({h: r["accuracy"] * 100 for h, r in results.items()})
    st.bar_chart(acc_series)

# =============================================================================
# HALAMAN 2: DETAIL PER HOTEL
# =============================================================================
elif halaman == "Detail per Hotel":
    hotel_pilihan = st.selectbox("Pilih Hotel", list(CSV_FILES.keys()))
    df = clean_data[hotel_pilihan]
    res = results[hotel_pilihan]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Komentar", len(df))
    col2.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
    col3.metric("Sentimen Dominan", df["label"].value_counts().idxmax())

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distribusi Sentimen", "Confusion Matrix", "Top Fitur TF-IDF", "Data Komentar"]
    )

    with tab1:
        vc = df["label"].value_counts().reindex(["Positif", "Negatif", "Netral"], fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(vc.index, vc.values, color=[WARNA[l] for l in vc.index])
        for bar, val in zip(bars, vc.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Jumlah Komentar")
        ax.set_title(f"Distribusi Sentimen — {hotel_pilihan}")
        st.pyplot(fig)

        st.markdown("**Perbandingan Label Lama (InSet murni) vs Label Hybrid (baru)**")
        agree = (df["label"] == df["label_lama"]).mean() * 100
        st.write(f"Kesepakatan label lama vs baru: **{agree:.1f}%**")
        comp = pd.DataFrame({
            "Label Lama": df["label_lama"].value_counts(),
            "Label Baru": df["label"].value_counts(),
        }).fillna(0).astype(int)
        st.dataframe(comp, use_container_width=True)

    with tab2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                     xticklabels=res["labels"], yticklabels=res["labels"], ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title(f"Confusion Matrix — Accuracy {res['accuracy']*100:.2f}%")
        st.pyplot(fig)
        st.text("Classification Report:\n" + res["report_str"])

    with tab3:
        top10 = top_fitur_tfidf(res, n=10)
        top_df = pd.DataFrame(top10, columns=["Kata/Frasa", "Bobot TF-IDF"])
        st.dataframe(top_df, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=top_df.iloc[::-1], x="Bobot TF-IDF", y="Kata/Frasa", ax=ax, palette="viridis")
        st.pyplot(fig)

    with tab4:
        filter_label = st.multiselect(
            "Filter berdasarkan label", ["Positif", "Netral", "Negatif"],
            default=["Positif", "Netral", "Negatif"],
        )
        tampil = df[df["label"].isin(filter_label)][["komentar_asli", "komentar", "label_lama", "label"]]
        st.dataframe(tampil, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Unduh CSV hasil relabel hotel ini",
            tampil.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{hotel_pilihan.replace(' ', '_')}_relabel.csv",
            mime="text/csv",
        )

# =============================================================================
# HALAMAN 3: COBA LABEL SENDIRI (interaktif)
# =============================================================================
else:
    from pipeline import unduh_inset, bangun_leksikon_hybrid, labeling_hybrid

    st.subheader("🧪 Coba Label Komentar Sendiri")
    st.write("Ketik komentar (gaya TikTok, boleh tidak baku) untuk melihat label sentimennya.")

    contoh = st.selectbox(
        "Contoh cepat",
        ["(ketik sendiri)", "kamar bagus tapi kotor", "pernah situ",
         "hotel nya bagus banget pelayanan ramah", "berapa harganya?",
         "pengalaman horor kapok situ takut"],
    )
    default_text = "" if contoh == "(ketik sendiri)" else contoh
    teks = st.text_area("Komentar", value=default_text, height=100)

    if st.button("Analisis Sentimen"):
        raw_pos, raw_neg = unduh_inset()
        kata_positif, kata_negatif = bangun_leksikon_hybrid(raw_pos, raw_neg)
        label = labeling_hybrid(teks, kata_positif, kata_negatif)
        warna_label = WARNA.get(label, "#999999")
        st.markdown(
            f"<h3 style='color:{warna_label}'>Label: {label}</h3>",
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption(
    "Tugas Akhir — Hikmal, NIM 3202302015, D3 Manajemen Informatika, "
    "Politeknik Negeri Sambas, 2026"
)
