import streamlit as st
import pandas as pd

# Load your merged dataset
@st.cache_data
def load_data():
    return pd.read_csv("E:\DIV FOLDER\project_samarth\data\merged_agri_data.csv") 
merged_df = load_data()


from query_engine import classify_query, match_crop, match_state, parse_year, validate_query, answer_query
# Streamlit UI
st.set_page_config(page_title="Samarth - Agri Q&A", layout="centered")
st.title("🌾 Samarth: Ask About Agriculture Data")

user_query = st.text_input("Enter your question:", placeholder="e.g. What was the rice yield in Tamil Nadu in 2021?")

if user_query:
    with st.spinner("Thinking..."):
        response = answer_query(user_query, merged_df)
    st.success("Answer:")
    st.write(response)