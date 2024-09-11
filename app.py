import streamlit as st
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
print(tf.__version__)

# 1. Load the model
# model = pickle.load(open('model_log.pkl', 'rb'))
model = tf.keras.models.load_model('model_ann_hamilelik.keras')

# 2. Load the StandardScaler that was used during training
scaler = pickle.load(open('scaler.pkl', 'rb'))


# 3. Title and description
st.title("Daxil etdiyiniz məlumatlara əsasən Hamiləlik proqnozu")
st.write("Bu tətbiq sinir şəbəkəsi modeli ilə proqnoz verir. Aşağıda dəyərləri daxil edin və 'Proqnoz' düyməsinə basın.")

# 4. Collect user inputs (fields for entering data)

# Header
st.header('Ginekoloji Xəstəliklər və Sağlamlıq Məlumatları')

# Columns for inputs
col1, col2, col3, col4 = st.columns(4)

# Inputs in the first column
with col1:
    yas = st.number_input('Yaş', min_value=0, max_value=120, step=1)
    bmi = st.number_input('BMI', min_value=0.0, format="%.1f")
    
# Inputs in the second column
with col2:
    fsh = st.number_input('FSH', min_value=0.0, format="%.1f")
    amh = st.number_input('AMH', min_value=0.0, format="%.1f")

# Inputs in the third column
with col3:
    hamilelik_sayisi = st.number_input('Hamiləlik sayısı', min_value=0, step=1)
    dogus_sayisi = st.number_input('Doğuş sayısı', min_value=0, step=1)

# Inputs in the fourth column
with col4:
    sonsuzluq_suresi = st.number_input('Sonsuzluq müddəti (ay)', min_value=0, step=1)
    endometrial_qalinliq = st.number_input('Endometrial qalınlıq (mm)', min_value=0.0, format="%.1f")


# Checkboxes for gynecological conditions

st.subheader('Ginekoloji Xəstəliklər')

# "Yoxdur" checkbox
yoxdur = st.checkbox('Yoxdur')

# Columns for gynecological conditions checkboxes
col1, col2, col3 = st.columns(3)

# Disable other checkboxes if "Yoxdur" is selected
disabled = False
if yoxdur:
    disabled = True

# Checkboxes for different conditions in columns
with col1:
    endometrit = st.checkbox('Endometrit', disabled=disabled)
    kista = st.checkbox('Kista', disabled=disabled)
    mioma = st.checkbox('Mioma', disabled=disabled)


with col2:
    polip = st.checkbox('Polip', disabled=disabled)
    salpiqoofrit = st.checkbox('Salpiqoofrit', disabled=disabled)
    emeliyyat = st.checkbox('Əməliyyat', disabled=disabled)

with col3:
    ub_patologiya = st.checkbox('UB patologiya', disabled=disabled)
    bakterial_vaginoz = st.checkbox('Bakterial vaginoz', disabled=disabled)
    usakliq_anomaliyasi = st.checkbox('Uşaqlıq anomaliyası', disabled=disabled)

    
# If "None" is selected, disable other options
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


# Sonsuzluq növləri section
st.subheader('Sonsuzluq Növləri')

# Sonsuzluq növləri Radio button for

sonsuzluq_novu = st.radio("Sonsuzluq növü seçin:", ('Birincili', 'Ikincili'))
if sonsuzluq_novu=='Birincili':
    birincili = True
    ikincili = False
else:
    ikincili = True
    birincili = False


# Sonsuzluq diaqnozları section
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



# Show the entered information
st.subheader("Sizin Məlumatlarınız:")


# Collect the input data into a dictionary
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

# Convert the dictionary into a pandas DataFrame and display it
df = pd.DataFrame(data)
st.write(df)

# Create a dictionary with the selected values for Ginekoloji Xəstəliklər, Sonsuzluq Diaqnozları, and Sonsuzluq Növləri
# Create a dictionary with the selected values
data = {
    "Ginekoloji Xəstəliklər": [],
    "Sonsuzluq Diaqnozları": [],
    "Sonsuzluq Növləri": []
}

# Add conditions to the dictionary based on the checkboxes
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

# Add Sonsuzluq Diaqnozları to the dictionary
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

# Add Sonsuzluq Növləri to the dictionary
if birincili:
    data["Sonsuzluq Növləri"].append("Birincili")
elif ikincili:
    data["Sonsuzluq Növləri"].append("İkincili")

# Convert the dictionary into a DataFrame and display it
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

# Replace NaN values with empty strings
df = df.fillna('')

# Display the DataFrame
st.write(df)



# 5. Prediction button
if st.button('Proqnoz'):
    # 6. Prepare the data for prediction
    input_data = np.array([[yas, bmi, fsh, amh, hamilelik_sayisi, dogus_sayisi, sonsuzluq_suresi, endometrial_qalinliq,
                            bakterial_vaginoz, endometrit, kista, mioma, polip, salpiqoofrit, ub_patologiya, 
                            usakliq_anomaliyasi, yoxdur, emeliyyat, birincili, ikincili, az_yumurta_ehtiyyati, 
                            boru_faktoru, digər, kisi_faktoru, pkys]])
    
    # Scale the data
    input_data_scaled = scaler.transform(input_data)
    
    # 7. Make a prediction
    # prediction = model.predict_proba(input_data_scaled)        # for Log model
    # predict_percent = prediction[0,1]*100

    prediction = model.predict(input_data_scaled)            # for ANN model
    predict_percent = float(prediction[0]) * 100
    
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
            <h2 style="color: #4CAF50;">☹️☹️☹️☹️☹️</h2>
            <p style="font-size: 24px; color: #9d00ff;">Ehtimalınız:</p>
            <h1 style="color: #93C572;">{predict_percent:.1f}%</h1>
        </div>
        """

    # Display the HTML in Streamlit
    st.markdown(card_html, unsafe_allow_html=True)

