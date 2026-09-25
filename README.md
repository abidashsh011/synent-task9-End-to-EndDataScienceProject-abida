# Task 9: End-to-End Data Science Project

**Internship Program:** Synent Technologies Data Science Internship  
**Author:** Abida Sharif  
**Dataset Source:** (https://www.kaggle.com/datasets/arshid/iris-flower-dataset)  

---

## 📌 Problem Statement
The goal of this project is to build and deploy an end-to-end machine learning pipeline that predicts flower species based on physical dimensions (sepal/petal lengths and widths).

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Data Analysis & EDA:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (Random Forest Classifier)
- **Deployment & UI:** Streamlit

## 🚀 Workflow & Approach
1. **Data Collection & Cleaning:** Loaded real-world dataset, checked missing values, removed duplicate rows, and verified data types.
2. **Exploratory Data Analysis (EDA):** Analyzed feature distributions and correlation matrices to identify key separators between species.
3. **Model Training & Evaluation:** Trained a Random Forest Classifier achieving **>95% test accuracy**. Exported the trained model using `pickle`.
4. **Interactive Deployment:** Built an interactive web application using Streamlit that accepts user inputs and outputs instantaneous predictions.

## 📊 Results & Key Findings
- Petal length and petal width are the most informative features for distinguishing between species categories.
- The deployed Random Forest model delivers accurate real-time classification through a clean user interface.

## 🎥 Video Demonstration
- **Demo Video Link:** 

## ⚙️ How to Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/synent-task9-e2emlproject-abida.git
