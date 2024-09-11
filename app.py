import streamlit as st
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 1. Modeli yükləyin
#model = pickle.load(open('model_log.pkl', 'rb'))
model = tf.keras.models.load_model('model_ann_hamilelik.keras')

# 2. StandardScaler üçün nümunə: təlim zamanı istifadə etdiyiniz scaler-i yenidən yükləməli və ya saxlamalısınız.
scaler = pickle.load(open('scaler.pkl', 'rb'))


# 3. Başlıq və izahat
st.title("Daxil etdiyiniz məlumatlara əsasən Hamiləlik proqnozu")
st.write("Bu tətbiq sinir şəbəkəsi modeli ilə proqnoz verir. Aşağıda dəyərləri daxil edin və 'Proqnoz' düyməsinə basın.")

# 4. İstifadəçi girişini tələb edin (məlumatları daxil etmək üçün sahələr)

# Başlıq
st.header('Ginekoloji Xəstəliklər və Sağlamlıq Məlumatları')

# Sütunlar
col1, col2, col3, col4 = st.columns(4)

# İlk sütunda yerləşən inputlar
with col1:
    yas = st.number_input('Yaş', min_value=0, max_value=120, step=1)
    bmi = st.number_input('BMI', min_value=0.0, format="%.1f")
    
# İkinci sütunda yerləşən inputlar
with col2:
    fsh = st.number_input('FSH', min_value=0.0, format="%.1f")
    amh = st.number_input('AMH', min_value=0.0, format="%.1f")

# Üçüncü sütunda yerləşən inputlar
with col3:
    hamilelik_sayisi = st.number_input('Hamiləlik sayısı', min_value=0, step=1)
    dogus_sayisi = st.number_input('Doğuş sayısı', min_value=0, step=1)

with col4:
    sonsuzluq_suresi = st.number_input('Sonsuzluq müddəti (ay)', min_value=0, step=1)
    endometrial_qalinliq = st.number_input('Endometrial qalınlıq (mm)', min_value=0.0, format="%.1f")


# Checkbox-lar

st.subheader('Ginekoloji Xəstəliklər')

# "Yoxdur" seçimi
yoxdur = st.checkbox('Yoxdur')

# Sütunlar
col1, col2, col3 = st.columns(3)

# Əgər "Yoxdur" seçilibsə, digər checkbox-ları deaktiv et
disabled = False
if yoxdur:
    disabled = True

# İlk sütunda yerləşən checkbox-lar
with col1:
    endometrit = st.checkbox('Endometrit', disabled=disabled)
    kista = st.checkbox('Kista', disabled=disabled)
    mioma = st.checkbox('Mioma', disabled=disabled)

# İkinci sütunda yerləşən checkbox-lar
with col2:
    polip = st.checkbox('Polip', disabled=disabled)
    salpiqoofrit = st.checkbox('Salpiqoofrit', disabled=disabled)
    emeliyyat = st.checkbox('Əməliyyat', disabled=disabled)

with col3:
    ub_patologiya = st.checkbox('UB patologiya', disabled=disabled)
    bakterial_vaginoz = st.checkbox('Bakterial vaginoz', disabled=disabled)
    usakliq_anomaliyasi = st.checkbox('Uşaqlıq anomaliyası', disabled=disabled)

    
# 'Yoxdur' seçildikdə digər seçimləri deaktiv edin
if yoxdur:
    st.warning('“Yoxdur” seçildikdə digər seçimlər deaktivdir.')
    bakterial_vaginoz = False
    endometrit = False
    kista = False
    mioma = False
    polip = False
    salpiqoofrit = False
    ub_patologiya = False
    usakliq_anomaliyasi = False
    emeliyyat = False


# Sonsuzluq növləri başlığı
st.subheader('Sonsuzluq Növləri')

# Sonsuzluq növləri üçün radio buttom

sonsuzluq_novu = st.radio("Sonsuzluq növü seçin:", ('Birincili', 'Ikincili'))
if sonsuzluq_novu=='Birincili':
    birincili = True
    ikincili = False
else:
    ikincili = True
    birincili = False


# Sonsuzluq diaqnozları
st.subheader('Sonsuzluq Diaqnozları')

col1, col2, col3 = st.columns(3)

with col1:
    az_yumurta_ehtiyyati = st.checkbox('Az yumurta ehtiyyatı')
    boru_faktoru = st.checkbox('Boru faktoru')
with col2:
    kisi_faktoru = st.checkbox('Kişi faktoru')
    pkys = st.checkbox('PKYS')
with col3:
   digər = st.checkbox('Digər')



# Nəticələri göstər
st.subheader("Sizin Məlumatlarınız:")


# Verilənləri bir sözlük (dictionary) formatında yığırıq
data = {
    "Yaş": [yas],
    "BMI": [bmi],
    "FSH": [fsh],
    "AMH": [amh],
    "Hamiləlik sayı": [hamilelik_sayisi],
    "Doğuş sayı": [dogus_sayisi],
    "Sonsuzluq müddəti": f'{sonsuzluq_suresi} ay',
    "Endometrial qalınlıq": f'{endometrial_qalinliq} mm'
}

# Sözlükdən pandas DataFrame yaradırıq
df = pd.DataFrame(data)

# DataFrame-i Streamlit vasitəsilə göstəririk
st.write(df)

# Create a dictionary with the selected values
data = {
    "Ginekoloji Xəstəliklər": [],
    "Sonsuzluq Diaqnozları": [],
    "Sonsuzluq Növləri": []
}

# Ginekoloji Xəstəliklər
if bakterial_vaginoz:
    data["Ginekoloji Xəstəliklər"].append("Bakterial vaginoz")
if endometrit:
    data["Ginekoloji Xəstəliklər"].append("Endometrit")
if kista:
    data["Ginekoloji Xəstəliklər"].append("Kista")
if mioma:
    data["Ginekoloji Xəstəliklər"].append("Mioma")
if polip:
    data["Ginekoloji Xəstəliklər"].append("Polip")
if salpiqoofrit:
    data["Ginekoloji Xəstəliklər"].append("Salpiqoofrit")
if ub_patologiya:
    data["Ginekoloji Xəstəliklər"].append("UB patologiya")
if usakliq_anomaliyasi:
    data["Ginekoloji Xəstəliklər"].append("Uşaqlıq anomaliyası")
if emeliyyat:
    data["Ginekoloji Xəstəliklər"].append("Əməliyyat")

if yoxdur:
    data["Ginekoloji Xəstəliklər"] = ["Yoxdur"]

# Sonsuzluq Diaqnozları
if az_yumurta_ehtiyyati:
    data["Sonsuzluq Diaqnozları"].append("Az yumurta ehtiyyatı")
if boru_faktoru:
    data["Sonsuzluq Diaqnozları"].append("Boru faktoru")
if digər:
    data["Sonsuzluq Diaqnozları"].append("Digər")
if kisi_faktoru:
    data["Sonsuzluq Diaqnozları"].append("Kişi faktoru")
if pkys:
    data["Sonsuzluq Diaqnozları"].append("PKYS")

# Sonsuzluq Növləri
if birincili:
    data["Sonsuzluq Növləri"].append("Birincili")
elif ikincili:
    data["Sonsuzluq Növləri"].append("İkincili")

# Create a DataFrame
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

# Replace NaN values with empty strings
df = df.fillna('')

# Display the DataFrame
st.write(df)



# 5. Proqnoz üçün düymə
if st.button('Proqnoz'):
    # 6. İstifadəçi girişindən alınan dəyərləri proqnoz üçün hazırlayın
    input_data = np.array([[yas, bmi, fsh, amh, hamilelik_sayisi, dogus_sayisi, sonsuzluq_suresi, endometrial_qalinliq,
                            bakterial_vaginoz, endometrit, kista, mioma, polip, salpiqoofrit, ub_patologiya, 
                            usakliq_anomaliyasi, yoxdur, emeliyyat, birincili, ikincili, az_yumurta_ehtiyyati, 
                            boru_faktoru, digər, kisi_faktoru, pkys]])
    
    # Veriləri uyğun şəkildə transformasiya edin (scaler təlimdən saxlanılmış olmalıdır)
    input_data_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_data_scaled)
    predict_percent = float(prediction[0]) * 100
    st.write(f"netice: {predict_percent}")
    
    # # 7. Proqnoz verin
    # prediction = model.predict_proba(input_data_scaled)
    # predict_percent = prediction[0,1]*100
    # # st.write(f'Sizin hamilə qalmaq ehtimalınız: {predict_percent:.1f} %')

    # Display the prediction probability
    if predict_percent > 80:
        card_html = f"""
        <div style="border: 2px solid #FFC107; border-radius: 10px; padding: 20px; text-align: center;">
            <h2 style="color: #4CAF50;">Təbriklər!</h2>
            <p style="font-size: 24px; color: #9d00ff;">🎉 Hər şey mükəmməl görünür! Sizin hamilə qalmaq ehtimalınız:</p>
            <h1 style="color: #93C572;">{predict_percent:.1f}%</h1>
        </div>
        """

    elif 50 < predict_percent <= 80:
        card_html = f"""
        <div style="border: 2px solid #FFC107; border-radius: 10px; padding: 20px; text-align: center;">
            <h2 style="color: #4CAF50;">Yaxşı xəbər!</h2>
            <p style="font-size: 24px; color: #9d00ff;">😊 Sizin hamilə qalmaq ehtimalınız:</p>
            <h1 style="color: #93C572;">{predict_percent:.1f}%</h1>
        </div>
        """

    elif 30 < predict_percent <= 50:
        card_html = f"""
        <div style="border: 2px solid #FFC107; border-radius: 10px; padding: 20px; text-align: center;">
            <h2 style="color: #4CAF50;">Normal</h2>
            <p style="font-size: 24px; color: #9d00ff;">🌟 Göstəriciləriniz:</p>
            <h1 style="color: #93C572;">{predict_percent:.1f}%</h1>
        </div>
        """

    else:
        card_html = f"""
        <div style="border: 2px solid #FFC107; border-radius: 10px; padding: 20px; text-align: center;">
            <h2 style="color: #4CAF50;">Dəstəkləyici Mesaj</h2>
            <p style="font-size: 24px; color: #9d00ff;">💭 Ehtimalınız:</p>
            <h1 style="color: #93C572;">{predict_percent:.1f}%</h1>
        </div>
        """

    # Display the HTML in Streamlit
    st.markdown(card_html, unsafe_allow_html=True)

