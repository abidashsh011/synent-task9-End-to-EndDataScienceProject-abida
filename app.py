import streamlit as st
import numpy as np
import pickle
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

model_path = Path(__file__).resolve().parent / 'model.pkl'
if model_path.exists():
    with model_path.open('rb') as f:
        model = pickle.load(f)
else:
    iris = load_iris()
    model = RandomForestClassifier(random_state=42)
    model.fit(iris.data, iris.target)

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
