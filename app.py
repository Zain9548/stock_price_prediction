import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# 🎯 Page setup
st.set_page_config(page_title="📈 Stock Price Predictor", page_icon="💹", layout="centered")

st.title("📊 Stock Price Prediction using Saved Linear Regression Model")
st.markdown("---")

# 🧠 Load the trained model
with open("model (3).pkl", "rb") as f:
    model = pickle.load(f)

st.success("✅ Model loaded successfully!")

# 📂 Upload stock data
uploaded_file = st.file_uploader("📁 Upload your stock CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    try:
        X = df[['Open', 'High', 'Low', 'Volume']]
    except KeyError:
        st.error("⚠️ CSV must have columns: Open, High, Low, Volume")
        st.stop()

    # 🧮 Predict button
    if st.button("🔮 Predict Stock Prices"):
        y_pred = model.predict(X)
        df['Predicted_Close'] = y_pred

        st.subheader("📈 Predicted Stock Prices (in table)")
        st.dataframe(df[['Open', 'High', 'Low', 'Volume', 'Predicted_Close']].head(20))

        # 📊 Graph plot
        st.subheader("📉 Predicted Price Trend")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_pred, label='Predicted Price', color='#FF6F00', linewidth=2)
        ax.set_xlabel("Days")
        ax.set_ylabel("Predicted Close Price")
        ax.set_title("Predicted Stock Closing Price Trend")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
else:
    st.info("👈 Please upload a stock CSV file to start prediction.")

st.markdown("---")
st.caption("Built with ❤️ by Mohammed Azeem")
