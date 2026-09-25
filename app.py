import streamlit as st
import numpy as np
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🌺 Iris Flower Species Predictor")

sepal_l = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_w = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_l = st.slider("Petal Length (cm)", 1.0, 7.0, 4.35)
petal_w = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

if st.button("Predict"):
    features = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    prediction = model.predict(features)[0]
    target_names = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Predicted Species: **{target_names[prediction]}**")