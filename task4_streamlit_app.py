"""Task 4: Interactive Streamlit Titanic Survival Predictor."""

import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

@st.cache_resource
def train_model():
    df = pd.read_csv(URL)
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    X, y = df[features], df["survived"]

    numeric = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical = ["sex", "embarked"]

    prep = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    model = Pipeline([
        ("preprocessor", prep),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X, y)
    return model

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger information to estimate the probability of survival.")

pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Siblings / spouses aboard", 0, 8, 0)
parch = st.number_input("Parents / children aboard", 0, 6, 0)
fare = st.number_input("Fare", min_value=0.0, value=32.20, step=1.0)
embarked = st.selectbox("Port of embarkation", ["S", "C", "Q"])

if st.button("Predict survival"):
    model = train_model()
    sample = pd.DataFrame([{
        "pclass": pclass, "sex": sex, "age": age,
        "sibsp": sibsp, "parch": parch, "fare": fare,
        "embarked": embarked
    }])
    probability = model.predict_proba(sample)[0, 1]
    prediction = model.predict(sample)[0]

    if prediction == 1:
        st.success(f"Prediction: Survived — probability {probability:.1%}")
    else:
        st.error(f"Prediction: Did not survive — survival probability {probability:.1%}")
