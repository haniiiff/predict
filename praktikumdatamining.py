import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Import data
data = pd.read_csv("data.csv")

# Sidebar untuk memilih tanggal data
st.sidebar.title("Pilih Tanggal Data")
selected_year = st.sidebar.slider("Tahun Data", 2020, 2021, step=1)
selected_half = st.sidebar.selectbox("Per Tiga Bulan", ["1st Half", "2nd Half", "3nd Half"])

# Fungsi untuk mengubah format tanggal
def format_date(year, half):
    if half == "1st Half":
        month = "03"
    elif half == "2nd Half" and year == 2021:
        month = "06"
    elif half == "3nd Half" and year == 2020:
        month = "12"
    else:
        month = "12"
    return f"{year}-{month}"

# Mengubah format tanggal yang dipilih oleh pengguna
selected_date = format_date(selected_year, selected_half)

# Fungsi untuk membersihkan karakter pada kolom 'Users'
def clean_users(x):
    if 'K' in x:
        return float(x.replace('K', '')) * 1000
    elif 'M' in x:
        return float(x.replace('M', '')) * 1000000
    else:
        return float(x)

# Menggunakan fungsi clean_users untuk membersihkan kolom 'Users'
data['Users'] = data['Users'].apply(clean_users)
# Menghapus tanda koma pada kolom 'Population' dan mengubah tipe datanya ke float
data['Population'] = data['Population'].str.replace(',', '')


def predict_user(country_name, population, user_percentage):
    X = population.reshape(-1,1)
    y = user_percentage.reshape(-1,1)
    if X.shape != y.shape:
        y = y.reshape(-1,1)
    model = LinearRegression()
    model.fit(X,y)
    return np.round(model.predict(X)[0][0])

# Callback function untuk tombol "Predict"
def on_predict_button_clicked():
    # Mengambil data yang sesuai dengan tanggal data
    filtered_data = data[data['Date_of_Data'] == selected_date]
    filtered_data['Population'] = filtered_data['Population'].apply(clean_users)

    # Menampilkan pesan error jika data tidak ditemukan
    if filtered_data.empty:
        st.error("Data tidak ditemukan untuk tanggal yang dipilih. Silakan pilih tanggal lain.")
        return

    # Mencari negara dengan pengguna Facebook terbanyak
    max_user = filtered_data['Users'].max()
    max_country = filtered_data.loc[filtered_data['Users']==max_user]['Name'].values[0]

    # Mencari negara dengan pengguna Facebook terendah
    min_user = filtered_data['Users'].min()
    min_country = filtered_data.loc[filtered_data['Users']==min_user]['Name'].values[0]

    # Menampilkan hasil pada main page
    st.title("Prediksi Jumlah Pengguna Facebook Terbanyak dan Terendah")
    st.write("Tanggal Data: ", filtered_data['Date_of_Data'].iloc[0])

    # Menampilkan hasil prediksi untuk negara dengan pengguna terbanyak
    st.write("Negara dengan Pengguna Facebook Terbanyak: ", max_country)
    X_max = filtered_data.loc[filtered_data['Name']==max_country, 'Population'].values
    y_max = filtered_data.loc[filtered_data['Name']==max_country, 'Users'].values
    pred_max = predict_user(max_country, X_max, y_max)
    st.write("Prediksi Jumlah Pengguna Facebook: ", pred_max, " pengguna")

    # Menampilkan hasil prediksi untuk negara dengan pengguna terendah
    st.write("Negara dengan Pengguna Facebook Terendah: ", min_country)
    X_min = filtered_data.loc[filtered_data['Name']==min_country, 'Population'].values
    y_min = filtered_data.loc[filtered_data['Name']==min_country, 'Users'].values
    pred_min = predict_user(min_country, X_min, y_min)
    st.write("Prediksi Jumlah Pengguna Facebook: ", pred_min, " pengguna")

    st.write("Data yang digunakan:")
    st.dataframe(filtered_data)

st.image("Logo Campus.png", width=150)
st.title("Tugas Akhir Praktikum Data Mining")
st.write("Faizal Ridho - 312210682")

predict_button = st.button("Predict")
if predict_button:
    on_predict_button_clicked()