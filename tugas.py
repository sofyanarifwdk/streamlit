import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso

st.title('Aplikasi Prediksi Tagihan Listrik Rumah')

# Penjelasan Model
st.subheader("Tentang Model Prediksi")
st.markdown("""
Model ini menggunakan algoritma regresi seperti **Linear Regression**, **Ridge**, dan **Lasso** untuk memprediksi tagihan listrik rumah tangga berdasarkan jumlah penggunaan listrik (dalam kWh).

Dengan memasukkan nilai kWh, pengguna akan mendapatkan estimasi jumlah tagihan listrik.

""")

# Pilihan algoritma
st.subheader("Pilih Algoritma Regresi")
model_choice = st.selectbox('Pilih Algoritma', ['Linear Regression', 'Ridge', 'Lasso'])

# Upload dataset
st.subheader("Upload Dataset (CSV)")
uploaded_file = st.file_uploader("Upload file CSV dengan kolom 'kWh' dan 'Tagihan'", type='csv')

# Gunakan data dari file upload atau data default
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil diupload!")
else:
    data = {
        'kWh': [100, 150, 200, 250, 300, 120, 180, 220, 160, 280],
        'Tagihan': [150000, 225000, 300000, 375000, 450000, 180000, 270000, 330000, 240000, 420000]
    }
    df = pd.DataFrame(data)
    st.info("Menggunakan dataset default.")

# Input Penggunaan
st.subheader("Input Penggunaan Listrik")
kwh = st.number_input('Masukkan Penggunaan Listrik (kWh)', min_value=0.0, step=0.1)

# Pemodelan
X = df[['kWh']]
y = df['Tagihan']

if model_choice == 'Linear Regression':
    model = LinearRegression()
elif model_choice == 'Ridge':
    model = Ridge(alpha=1.0)
else:  # Lasso
    model = Lasso(alpha=1.0)

model.fit(X, y)

# Prediksi
prediksi = None
if st.button('Prediksi Tagihan'):
    prediksi = model.predict(np.array([[kwh]]))[0]
    st.success(f'Prediksi Tagihan Listrik: Rp {prediksi:,.0f}')

# Tampilkan dataset
st.subheader("Dataset")
st.write(df)

# Visualisasi
st.subheader("Visualisasi Data dan Garis Regresi")
fig, ax = plt.subplots()
ax.scatter(df['kWh'], df['Tagihan'], color='blue', label='Data Asli')

x_range = np.linspace(df['kWh'].min(), df['kWh'].max(), 100).reshape(-1, 1)
y_pred = model.predict(x_range)
ax.plot(x_range, y_pred, color='red', label='Garis Regresi')

# Titik prediksi
if prediksi is not None:
    ax.scatter(kwh, prediksi, color='green', s=100, label='Prediksi Anda', zorder=5)

ax.set_xlabel('Penggunaan Listrik (kWh)')
ax.set_ylabel('Tagihan Listrik (Rp)')
ax.legend()
st.pyplot(fig)
