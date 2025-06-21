import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Mobile Dataset Analysis", layout="wide")
st.title("📱 Mobile Dataset Analysis (2025)")

uploaded_file = st.file_uploader(
    "Upload your Mobile Dataset CSV file", type=["csv"]
)

@st.cache_data
def load_and_clean_data(file):
    df = pd.read_csv(file, encoding='ISO-8859-1')

    currency_symbols = {
        'PKR': '', 'INR': '', 'CNY': '', 'USD': '', 'AED': '',
        '₹': '', '¥': '', '$': '', 'د.إ': '', ',': '', 'Not available': np.nan
    }
    price_columns = [
        'Launched Price (Pakistan)', 'Launched Price (India)',
        'Launched Price (China)', 'Launched Price (USA)', 'Launched Price (Dubai)'
    ]
    for col in price_columns:
        df[col] = df[col].replace(currency_symbols, regex=True).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    def clean_numeric_column(col, remove_str):
        cleaned = df[col].fillna('').astype(
            str).str.replace(remove_str, '', regex=False)
        cleaned = cleaned.replace('Not available', np.nan)
        return pd.to_numeric(cleaned, errors='coerce')

    df['Mobile Weight'] = clean_numeric_column('Mobile Weight', 'g')
    df['Battery Capacity'] = clean_numeric_column('Battery Capacity', 'mAh')
    df['Screen Size'] = clean_numeric_column('Screen Size', 'inches')
    df['RAM'] = clean_numeric_column('RAM', 'GB')

    df.dropna(subset=price_columns +
              ['Mobile Weight', 'Battery Capacity', 'Screen Size', 'RAM'], inplace=True)
    return df


if uploaded_file is not None:
    try:
        df = load_and_clean_data(uploaded_file)
        st.success("✅ Data loaded and cleaned!")

        st.subheader("Data Preview")
        st.dataframe(df.head())

        st.subheader("Battery Capacity Distribution for Top 5 Companies")
        top_5_companies = df['Company Name'].value_counts().head(5).index
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(
            data=df[df['Company Name'].isin(top_5_companies)],
            x='Company Name', y='Battery Capacity', ax=ax
        )
        ax.set_ylabel('Battery Capacity (mAh)')
        ax.set_xlabel('Company Name')
        st.pyplot(fig)

        st.subheader("Average Screen Size by Launch Year")
        yearly_screen_size = df.groupby('Launched Year')[
            'Screen Size'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=yearly_screen_size, x='Launched Year',
                     y='Screen Size', marker='o', color='purple', ax=ax)
        ax.set_ylabel('Average Screen Size (inches)')
        ax.set_xlabel('Launch Year')
        st.pyplot(fig)

        st.subheader("RAM Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df['RAM'], bins=10, kde=True, color="teal", ax=ax)
        ax.set_xlabel('RAM (GB)')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        st.subheader("Screen Size vs Battery Capacity")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df, x='Screen Size', y='Battery Capacity',
                        hue='Company Name', palette='Set2', legend=False, ax=ax)
        ax.set_xlabel('Screen Size (inches)')
        ax.set_ylabel('Battery Capacity (mAh)')
        st.pyplot(fig)

        # ================= PREDICTION SECTION =================
        st.header("📊 Predict Launched Price (India)")

        st.markdown("Enter the specifications below to predict the mobile's launch price in India.")

        with st.form("prediction_form"):
            ram = st.slider("RAM (GB)", 1, 24, step=1)
            screen_size = st.slider("Screen Size (inches)", 4.0, 7.5, step=0.1)
            battery = st.slider("Battery Capacity (mAh)", 1000, 7000, step=100)
            weight = st.slider("Mobile Weight (g)", 100, 300, step=10)
            submitted = st.form_submit_button("Predict Price")

        if submitted:
            # Model training (simple linear regression)
            X = df[['RAM', 'Screen Size', 'Battery Capacity', 'Mobile Weight']]
            y = df['Launched Price (India)']
            model = LinearRegression()
            model.fit(X, y)

            # Prediction
            input_features = np.array([[ram, screen_size, battery, weight]])
            predicted_price = model.predict(input_features)[0]
            st.success(f"💰 Predicted Launched Price (India): ₹{predicted_price:,.2f}")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("👆 Please upload a CSV file to get started.")
