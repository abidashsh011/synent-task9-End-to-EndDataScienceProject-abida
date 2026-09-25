import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Iris Species Predictor", page_icon="🌺")

# Load trained model artifact directly
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title("🌺 Iris Flower Species Predictor")

sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 4.3)
petal_width = st.number_input("Petal Width (cm)", 0.1, 2.5, 1.3)

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    species = {0: 'Setosa 🌸', 1: 'Versicolor 🌿', 2: 'Virginica 🌺'}
    st.success(f"Predicted Species: **{species.get(prediction, 'Unknown')}**")
