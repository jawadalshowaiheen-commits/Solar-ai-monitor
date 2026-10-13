import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Solar AI Monitor",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Solar AI Monitor")
st.subheader("Solar Power Data Analysis")

st.write(
    "Explore real solar power production and weather data."
)

@st.cache_data
def load_data():
    return pd.read_csv("Solar_Weather_Data_Processed.csv")

try:
    df = load_data()

    st.success("Solar dataset loaded successfully!")

    st.metric("Number of Records", f"{len(df):,}")
    st.metric("Number of Columns", len(df.columns))

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Available Data Columns")
    st.write(", ".join(df.columns.astype(str)))

    st.markdown("### Numerical Data Summary")
    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )

except Exception as e:
    st.error("Could not load the dataset.")
    st.write(str(e))
