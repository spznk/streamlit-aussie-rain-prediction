import streamlit as st
import joblib
import pandas as pd


def predict():
    # Extract inputs and target
    X = df[aus['input_cols']].copy()
    y = df[aus['target_col']].copy()

    # Run imputer
    cols_for_imputer = aus['imputer'].get_feature_names_out()
    X[cols_for_imputer] = aus['imputer'].transform(X[cols_for_imputer])

    # Run scaler
    num_cols = aus['numeric_cols']
    X[num_cols] = aus['scaler'].transform(X[num_cols])

    # Run encoder
    cat_cols = aus['categorical_cols']
    enc_cols = aus['encoded_cols']

    encoded_data = pd.DataFrame(
        aus['encoder'].transform(X[cat_cols]),
        columns=enc_cols,
        index=X.index
    )

    X = pd.concat(
        [X.drop(columns=cat_cols), encoded_data],
        axis=1
    )

    # Predict
    preds = aus['model'].predict(X)
    return preds


# Load model and other instruments from file
aus = joblib.load('aussie_rain_tree.joblib')

# Load data from file
df = pd.read_csv("weatherAUS.csv")

# Remove instances where 'RainToday' or 'RainTomorrow' are absent
df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

# App header
st.title("Rain prediction in Australia")
st.markdown("This model predicts whether there will be rain based on previous day data")

# Show table with mean values
st.header("Mean parameter values depending on target value")
mean_values = df.groupby(aus['target_col']).mean().reset_index()
st.dataframe(mean_values)