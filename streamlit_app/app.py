"""
app.py
=============================================================================
Streamlit App — Analisis Sentimen Hotel di Kota Singkawang
=============================================================================
"""

import sys
import os
from pathlib import Path

# 🔥 CRITICAL FIX: Tambahkan path project ke sys.path
# Ini penting untuk menemukan pipeline.py
BASE_DIR = Path(__file__).resolve().parent  # streamlit_app/
ROOT_DIR = BASE_DIR.parent                  # root project
sys.path.insert(0, str(BASE_DIR))           # tambahkan streamlit_app ke path
sys.path.insert(0, str(ROOT_DIR))           # tambahkan root ke path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Import dari pipeline.py (sekarang di folder yang sama dengan app.py)
try:
    from pipeline import (
        jalankan_pipeline, 
        top_fitur_tfidf, 
        ringkasan_distribusi_total,
        KATA_FILTER
    )
except ImportError as e:
    st.error(f"❌ Gagal import pipeline: {e}")
    st.info("Pastikan file pipeline.py berada di folder yang sama dengan app.py")
    st.stop()

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="Analisis Sentimen Hotel Singkawang",
    page_icon="📊",
    layout="wide",
)

# =============================================================================
# CARI FILE CSV
# =============================================================================
def cari_file_csv(nama_file):
    """Cari file CSV di beberapa lokasi yang mungkin"""
    lokasi = [
        os.path.join('data', nama_file),                     # streamlit_app/data/
        os.path.join('..', 'data', nama_file),               # ../data/
        os.path.join(ROOT_DIR, 'data', nama_file),           # root/data/
        os.path.join(BASE_DIR, 'data', nama_file),           # streamlit_app/data/
        nama_file,                                            # folder yang sama
    ]
    for path in lokasi:
        if os.path.exists(path):
            return path
    return None

CSV_NAMES = {
    "Hotel Mahkota Singkawang": "HOTEL_MAHKOTA_SINGKAWANG.csv",
    "Hotel Swiss-Belhotel Singkawang": "HOTEL_SWISS_BELHOTEL_SINGKAWANG.csv",
    "Hotel Dayang Resort Singkawang": "HOTEL_DAYANG_RESORT_SINGKAWANG.csv",
    "Hotel Horison Ultima Singkawang": "HOTEL_HORISON_ULTIMA_SINGKAWANG.csv",
}

CSV_FILES = {}
for hotel_name, fname in CSV_NAMES.items():
    path = cari_file_csv(fname)
    if path:
        CSV_FILES[hotel_name] = path
    else:
        # Fallback: gunakan path di streamlit_app/data/
        fallback_path = os.path.join(BASE_DIR, 'data', fname)
        CSV_FILES[hotel_name] = fallback_path
        st.warning(f"⚠️ File {fname} tidak ditemukan. Mencoba di {fallback_path}")

WARNA = {"Positif": "#4CAF50", "Negatif": "#F44336", "Netral": "#2196F3"}

st.title("📊 Analisis Sentimen Hotel di Kota Singkawang")
st.caption(
    "Implementasi Multinomial Naïve Bayes untuk Analisis Sentimen Komentar "
    "TikTok — Hybrid Lexicon Labeling (revisi pasca sidang)"
)

# =============================================================================
# CACHE: jalankan pipeline sekali saja
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def _run_pipeline():
    log_box = st.empty()
    logs = []

    def progress_cb(msg):
        logs.append(msg)
        log_box.info("\n\n".join(logs[-5:]))

    try:
        hasil = jalankan_pipeline(CSV_FILES, progress_cb=progress_cb)
        log_box.empty()
        return hasil
    except Exception as e:
        log_box.empty()
        st.error(f"❌ Error menjalankan pipeline: {e}")
        st.stop()


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
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**📌 Catatan Visualisasi TF-IDF**\n\n"
    "- Kata difilter untuk interpretasi\n"
    "- **TIDAK mengubah** perhitungan TF-IDF\n"
    "- **TIDAK mengubah** model & akurasi"
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Distribusi Sentimen", "Confusion Matrix", "Top Fitur TF-IDF", "Data Komentar", "Perbandingan Filter TF-IDF"]
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
        st.markdown("""
        **🔍 Top Fitur TF-IDF (Kata Informatif)**
        
        Kata-kata yang ditampilkan telah **difilter** dari kata tidak informatif 
        (nama kota, nama hotel, kata sambung, filler) agar lebih mudah diinterpretasikan.
        
        ⚠️ **Penting:** Penyaringan hanya untuk visualisasi, **TIDAK mengubah** 
        perhitungan TF-IDF, model, maupun metrik evaluasi.
        """)
        
        top_data = res.get('top_fitur', {})
        top_asli = top_data.get('top_asli', [])
        top_filtered = top_data.get('top_filtered', [])
        
        if top_filtered:
            col_a, col_b = st.columns([2, 3])
            with col_a:
                top_df = pd.DataFrame(top_filtered, columns=["Kata/Frasa", "Bobot TF-IDF"])
                st.dataframe(top_df, use_container_width=True, hide_index=True)
            with col_b:
                fig, ax = plt.subplots(figsize=(6, 4))
                top_df_plot = pd.DataFrame(top_filtered[::-1], columns=["Kata/Frasa", "Bobot TF-IDF"])
                sns.barplot(data=top_df_plot, x="Bobot TF-IDF", y="Kata/Frasa", ax=ax, palette="viridis")
                ax.set_title("Top Kata Informatif TF-IDF")
                st.pyplot(fig)
        else:
            st.info("Tidak ada kata yang lolos filter.")
        
        with st.expander("📋 Lihat Top TF-IDF Asli (tanpa filter)"):
            if top_asli:
                top_df_asli = pd.DataFrame(top_asli, columns=["Kata/Frasa", "Bobot TF-IDF"])
                top_df_asli['Status'] = top_df_asli['Kata/Frasa'].apply(
                    lambda x: '✕ Difilter' if x in KATA_FILTER else '✅ Dipertahankan'
                )
                st.dataframe(top_df_asli, use_container_width=True, hide_index=True)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#EF5350' if w in KATA_FILTER else '#42A5F5' 
                         for w, v in top_asli[::-1]]
                bars = ax.barh([w for w, v in top_asli[::-1]], 
                              [v for w, v in top_asli[::-1]], 
                              color=colors)
                ax.set_xlabel('Nilai TF-IDF')
                ax.set_title('Top TF-IDF Asli (Merah = akan difilter)')
                st.pyplot(fig)
            else:
                st.info("Tidak ada data top fitur asli.")

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

    with tab5:
        st.subheader("📊 Perbandingan Top TF-IDF Sebelum vs Sesudah Filter")
        st.markdown("""
        **Catatan Metodologis:**
        - Penyaringan kata dilakukan **HANYA UNTUK VISUALISASI** dan interpretasi.
        - **TIDAK mengubah** perhitungan TF-IDF, model, maupun metrik evaluasi.
        - Filter bertujuan membuang kata tidak informatif (nama kota, nama hotel, filler).
        """)
        
        top_data = res.get('top_fitur', {})
        top_asli = top_data.get('top_asli', [])[:10]
        top_filtered = top_data.get('top_filtered', [])[:10]
        
        if top_asli or top_filtered:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            if top_asli:
                words_asli = [w for w, v in top_asli]
                vals_asli = [v for w, v in top_asli]
                bars1 = axes[0].barh(words_asli[::-1], vals_asli[::-1], color='#B0BEC5')
                axes[0].set_title(f'{hotel_pilihan}\nSEBELUM Filter (Asli)', fontsize=10, fontweight='bold')
                axes[0].set_xlabel('Nilai TF-IDF')
                axes[0].grid(axis='x', linestyle='--', alpha=0.3)
                for j, (bar, word) in enumerate(zip(bars1, words_asli[::-1])):
                    if word in KATA_FILTER:
                        bar.set_color('#EF5350')
                        axes[0].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                                    '✕ (difilter)', va='center', fontsize=7, color='red')
            else:
                axes[0].text(0.5, 0.5, 'Tidak ada data', ha='center', va='center')
            
            if top_filtered:
                words_filtered = [w for w, v in top_filtered]
                vals_filtered = [v for w, v in top_filtered]
                bars2 = axes[1].barh(words_filtered[::-1], vals_filtered[::-1], color='#66BB6A')
                axes[1].set_title(f'{hotel_pilihan}\nSESUDAH Filter (Informatif)', fontsize=10, fontweight='bold')
                axes[1].set_xlabel('Nilai TF-IDF')
                axes[1].grid(axis='x', linestyle='--', alpha=0.3)
                for bar, val in zip(bars2, vals_filtered[::-1]):
                    axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                                f'{val:.2f}', va='center', fontsize=8)
            else:
                axes[1].text(0.5, 0.5, 'Tidak ada kata yang lolos filter', ha='center', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            filtered_words = [w for w, v in top_asli if w in KATA_FILTER]
            if filtered_words:
                with st.expander(f"📋 Kata yang difilter ({len(filtered_words)} kata)"):
                    kota_hotel = [w for w in filtered_words if w in ['singkawang', 'skw', 'mahkota', 'swiss', 'belhotel', 'dayang', 'horison', 'ultima', 'swissbell', 'resort', 'hotel']]
                    umum = [w for w in filtered_words if w not in kota_hotel]
                    if kota_hotel:
                        st.write(f"**Nama Kota/Hotel:** {', '.join(kota_hotel)}")
                    if umum:
                        st.write(f"**Kata Umum:** {', '.join(umum)}")
        else:
            st.info("Tidak ada data top fitur yang tersedia.")

# =============================================================================
# HALAMAN 3: COBA LABEL SENDIRI
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
        with st.spinner("Memproses..."):
            raw_pos, raw_neg = unduh_inset()
            kata_positif, kata_negatif = bangun_leksikon_hybrid(raw_pos, raw_neg)
            label = labeling_hybrid(teks, kata_positif, kata_negatif)
            warna_label = WARNA.get(label, "#999999")
            st.markdown(
                f"<h3 style='color:{warna_label}'>Label: {label}</h3>",
                unsafe_allow_html=True,
            )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption(
    "Tugas Akhir — Hikmal, NIM 3202302015, D3 Manajemen Informatika, "
    "Politeknik Negeri Sambas, 2026"
)
