import streamlit as st
import joblib
import pandas as pd


def predict(X: pd.DataFrame):
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
    return preds[0]


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
st.header("Mean numeric parameter values depending on 'RainTomorrow' value")

target_col = aus['target_col']
input_cols = aus['input_cols']
num_cols = aus['numeric_cols']
cat_cols = aus['categorical_cols']

mean_values = df[[*num_cols, target_col]].groupby(target_col).mean().reset_index()
st.dataframe(mean_values)

input_data = {}

# Header for parameters selection
st.header("Select parameters and click Predict button")

# Input parameters
col1, col2 = st.columns(2)

with col1:
    st.text("Number parameters")

    for col in num_cols:
        max_value = df[col].max()
        min_value = df[col].min()
        mean_value = df[col].mean()
        input_data[col] = [st.slider(col, min_value, max_value, mean_value)]
        

with col2:
    st.text("Selection parameters")

    for col in cat_cols:
        values = df[col].unique()
        input_data[col] = [st.pills(col, values, selection_mode="single")]

# Prediction button
if st.button("Will there be rain tomorrow?"):
    result = predict(pd.DataFrame(data=input_data))
    st.write(f"Answer: {result}.upper()")