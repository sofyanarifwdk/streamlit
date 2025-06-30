import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

st.title('Prediksi Tagihan Listrik Rumah Berbasis Regresi Linier')

st.header('Input Penggunaan Listrik')
kwh = st.number_input('Masukkan Penggunaan Listrik (kWh)', min_value=0.0, step=0.1)

# Dataset contoh
data = {
    'kWh': [100, 150, 200, 250, 300, 120, 180, 220, 160, 280],
    'Tagihan': [150000, 225000, 300000, 375000, 450000, 180000, 270000, 330000, 240000, 420000]
}
df = pd.DataFrame(data)

# Pemodelan regresi linier
X = df[['kWh']]
y = df['Tagihan']
model = LinearRegression()
model.fit(X, y)

# Prediksi
if st.button('Prediksi Tagihan'):
    prediksi = model.predict(np.array([[kwh]]))[0]
    st.success(f'Prediksi Tagihan Listrik: Rp {prediksi:,.0f}')

# Tampilkan dataset
st.subheader('Data Latih')
st.write(df)

# Visualisasi data dan garis regresi
st.subheader('Visualisasi Data dan Garis Regresi')

fig, ax = plt.subplots()
ax.scatter(df['kWh'], df['Tagihan'], color='blue', label='Data Asli')

# Garis regresi
x_range = np.linspace(df['kWh'].min(), df['kWh'].max(), 100)
y_pred = model.predict(x_range.reshape(-1, 1))
ax.plot(x_range, y_pred, color='red', label='Garis Regresi')

ax.set_xlabel('Penggunaan Listrik (kWh)')
ax.set_ylabel('Tagihan Listrik (Rp)')
ax.legend()

st.pyplot(fig)
