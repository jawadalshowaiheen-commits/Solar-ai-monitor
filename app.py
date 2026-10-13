import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Solar AI Monitor",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Solar AI Monitor")
st.subheader("Solar Power Prediction Using Machine Learning")
st.write("A machine-learning model trained on solar power and weather data.")

@st.cache_data
def load_data():
    return pd.read_csv("Solar_Weather_Data_Processed.csv")

@st.cache_resource
def train_model(data):
    features = [
        "GHI", "Temperature", "Relative Humidity",
        "Wind Speed", "Wind Direction", "Hour",
        "Day", "Month", "SZ Angle", "SA Angle"
    ]
    target = "AC Output"

    clean = data[features + [target]].copy()
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[clean[target] >= 0]

    X = clean[features]
    y = clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, features, clean, mae, r2, X_test, y_test, predictions

try:
    data = load_data()
    st.success(f"Dataset loaded: {len(data):,} records")

    model, features, clean, mae, r2, X_test, y_test, predictions = train_model(data)

    st.markdown("### Model Evaluation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records Used", f"{len(clean):,}")
    c2.metric("Mean Absolute Error", f"{mae:.2f}")
    c3.metric("Test R² Score", f"{r2:.3f}")

    st.caption(
        "The model predicts AC power output, not panel degradation. "
        "Test results may be optimistic because nearby hourly records "
        "can be similar."
    )

    st.markdown("### Try a Prediction")
    left, right = st.columns(2)
    defaults = clean[features].median()

    with st.form("prediction_form"):
        inputs = {}
        for i, feature in enumerate(features):
            col = left if i % 2 == 0 else right
            minimum = float(clean[feature].min())
            maximum = float(clean[feature].max())
            value = float(defaults[feature])
            if minimum < maximum:
                inputs[feature] = col.slider(
                    feature, minimum, maximum, value
                )
            else:
                inputs[feature] = col.number_input(
                    feature, value=value
                )
        submitted = st.form_submit_button("Predict Solar Power")

    if submitted:
        sample = pd.DataFrame([inputs], columns=features)
        result = max(0, float(model.predict(sample)[0]))
        st.success(f"Predicted AC Output: {result:.2f}")
        st.caption("Prediction is based on the uploaded dataset and is not a guarantee of real-world output.")

    st.markdown("### Actual vs Predicted (Test Sample)")
    chart = pd.DataFrame({
        "Actual AC Output": y_test.to_numpy()[:100],
        "Predicted AC Output": predictions[:100]
    })
    st.line_chart(chart)

except Exception as error:
    st.error("The model could not be trained. Check the data columns.")
    st.write(str(error))
