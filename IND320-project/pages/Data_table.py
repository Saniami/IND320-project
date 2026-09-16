import streamlit as st 
import pandas as pd
@st.cache_data
def load_data():
    data = pd.read_csv("data./reservoirs.csv")
    return data

df = load_data()
st.title("Reservoir Data Table")
