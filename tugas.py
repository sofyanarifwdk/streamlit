import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Judul Aplikasi
st.title('Aplikasi Prediksi Tagihan Listrik Rumah')

# Deskripsi Model
st.subheader("Tentang Model Prediksi")
st.markdown("""
Model ini menggunakan algoritma **Regresi Linier** untuk memprediksi jumlah tagihan listrik berdasarkan penggunaan listrik rumah tangga (dalam kWh).  
Model ini membentuk persamaan garis:
> **Tagihan = (koefisien × kWh) + intersep**
""")

# Input Penggunaan
st.subheader("Masukkan Penggunaan Listrik")
kwh = st.number_input("Masukkan Jumlah kWh", min_value=0.0, step=0.1)

# Upload CSV
st.subheader("Upload Dataset (opsional)")
uploaded_file = st.file_uploader("Upload file CSV dengan kolom 'kWh' dan 'Tagihan'", type="csv")

# Gunakan data dari file atau default
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset berhasil diupload.")
    except:
        st.error("Format CSV tidak sesuai.")
else:
    # Dataset Default
    data = {
        'kWh': [100, 150, 200, 250, 300, 120, 180, 220, 160, 280],
        'Tagihan': [150000, 225000, 300000, 375000, 450000, 180000, 270000, 330000, 240000, 420000]
    }
    df = pd.DataFrame(data)
    st.info("Menggunakan dataset default.")

# Tampilkan data
st.subheader("Dataset")
st.dataframe(df)

# Model Linear Regression
X = df[['kWh']]
y = df['Tagihan']
model = LinearRegression()
model.fit(X, y)

# Prediksi
prediksi = None
if st.button("Prediksi Tagihan"):
    prediksi = model.predict(np.array([[kwh]]))[0]
    st.success(f"Prediksi Tagihan Listrik: Rp {prediksi:,.0f}")

# Visualisasi
st.subheader("Visualisasi Garis Regresi")
fig, ax = plt.subplots()

# Data asli
ax.scatter(df['kWh'], df['Tagihan'], color='blue', label='Data Asli')

# Garis regresi
x_range = np.linspace(df['kWh'].min(), df['kWh'].max(), 100)
y_pred = model.predict(x_range.reshape(-1, 1))
ax.plot(x_range, y_pred, color='red', label='Garis Regresi')

# Titik prediksi
if prediksi is not None:
    ax.scatter(kwh, prediksi, color='green', s=100, label='Prediksi Anda', zorder=5)

ax.set_xlabel("Penggunaan Listrik (kWh)")
ax.set_ylabel("Tagihan Listrik (Rp)")
ax.legend()
st.pyplot(fig)
