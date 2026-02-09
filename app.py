import streamlit as st
import numpy as np
import joblib

model = joblib.load("pokemon_base_exp_model.pkl")

st.title("Predict EXP base Pokemon")
st.write("Evaluate Pokemon Strenght")

hp = st.number_input("HP", 1, 255)
attack = st.number_input("Attack", 1, 255)
defend = st.number_input("Defend", 1, 255)
spe = st.number_input("Special", 1, 255)
speed = st.number_input("Speed", 1, 255)

if st.button("Predict"):
    X = np.array([[hp, attack, defend, spe, speed]])
    pred = model.predict(X)[0]
    st.success(f"Estimated exp: {pred:.2f}")