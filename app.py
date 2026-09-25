import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import iris flower file
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Iris Species Predictor", page_icon="🌺", layout="centered")

# Train & Cache model directly inside Streamlit
@st.cache_resource
def get_trained_model():
    iris = load_iris()
    X = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    y = iris.target
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = get_trained_model()

st.title("🌺 Iris Flower Species Predictor")
st.write("Enter feature measurements below to predict the flower species.")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.3, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.3, step=0.1)

if st.button("Predict Species", type="primary"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    
    species_map = {0: 'Setosa 🌸', 1: 'Versicolor 🌿', 2: 'Virginica 🌺'}
    st.success(f"**Predicted Species:** {species_map.get(prediction, 'Unknown')}")
