import streamlit as st
import joblib
import pandas as pd


@st.cache_resource
def load_model():
    return joblib.load('aussie_rain_tree.joblib')


@st.cache_data
def load_data(_aus):
    # Load data from file
    df = pd.read_csv("weatherAUS.csv")

    # Remove instances where 'RainToday' or 'RainTomorrow' are absent
    df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

    # Run imputer
    cols = _aus['imputer'].get_feature_names_out()
    df[cols] = _aus['imputer'].transform(df[cols])

    return df

def predict(X: pd.DataFrame):
    # Ensure columns order
    X = X.loc[:, expected_cols]
    
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
aus = load_model()

target_col = aus['target_col']
input_cols = aus['input_cols']
num_cols = aus['numeric_cols']
cat_cols = aus['categorical_cols']

# Load data
df = load_data(aus)

# App header
st.title("Rain prediction in Australia")
st.markdown("This model predicts whether there will be rain tomorrow based on today's data")

# Show table with mean values
st.header("Mean numeric parameter values depending on 'RainTomorrow' value")

mean_values = df[[*num_cols, target_col]].groupby(target_col).mean().reset_index()
st.dataframe(mean_values)

# Header for parameters selection
st.header("Select parameters and click Predict button")

# Input parameters
col1, col2 = st.columns(2)
input_data = {}

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
        values = values[~pd.isna(values)]
        input_data[col] = [st.pills(col, values, selection_mode="single")]

# Prediction button
if st.button("Predict"):
    result = predict(pd.DataFrame(data=input_data))
    st.write(f"Rain tomorrow: {result.upper()}")