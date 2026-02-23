import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
import os

st.set_page_config(page_title="Soil Quality Detector")

# Load model
model = tf.keras.models.load_model("soil_model.h5")

API_KEY = "YOUR_OPENWEATHER_API_KEY"

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url).json()
        if response.get("main"):
            return response["main"]["temp"], response["main"]["humidity"]
    except:
        return None, None
    return None, None

st.title("🌱 Soil Quality Identification System")

uploaded_file = st.file_uploader("Upload Soil Image", type=["jpg", "png", "jpeg"])
city = st.text_input("Enter City Name")

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    classes = ["Construction Sand", "Agricultural Soil", "Clay Soil"]
    result = classes[np.argmax(prediction)]

    st.success(f"Predicted Soil Type: {result}")

    if city:
        temp, humidity = get_weather(city)
        if temp is not None:
            st.info(f"Temperature in {city}: {temp}°C")
            st.info(f"Humidity in {city}: {humidity}%")

            if humidity > 70:
                st.warning("High moisture may affect construction quality.")