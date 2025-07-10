import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Page config
st.set_page_config(page_title="𝗠𝗮𝗹𝗹𝗠𝗶𝗻𝗱𝘀  💼", page_icon="🛍️", layout="centered")
import base64

def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("c07a576a-5dca-4ce9-8cf6-9c003bbd6fca.png")


# Set your uploaded image as background
set_background("c07a576a-5dca-4ce9-8cf6-9c003bbd6fca.png")


# --- Custom CSS styling ---

st.markdown("""
    <style>
    .stApp {
        background-color: #fdf6f0;
        color: #000000 !important;
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h2, h3, h4, h5, h6, label, .css-1cpxqw2, .st-bc, .st-cz, .st-ag, .st-em {
        color: #663300 !important;
    }

    .stRadio > div {
        color: #000000 !important;
    }

    .stButton>button {
        background-color: #d9b99b;
        color: #000000;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #c89f84;
        color: brown;
    }

    .css-1d391kg p, .stSlider label {
        color: #663300 !important;
    }

    .stDataFrame {
        background-color: white !important;
        color: #663300 !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- App Title ---
st.markdown("<h1 style='text-align: center;'>🛍️ MallMinds – Customer Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #663300;'>Discover who shops what, and why.</h4>", unsafe_allow_html=True)

# --- Load & Prepare Data ---
df = pd.read_csv("Mall_Customers.csv")
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Train KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# --- Data Preview ---
st.markdown("<h3>🔍 Preview the Customer Data</h3>", unsafe_allow_html=True)
with st.expander("Click to view raw data"):
    st.dataframe(df.head())

# --- Cluster Plot ---
st.markdown("<h3>📊 Customer Segmentation</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots()
sns.set_style("whitegrid")
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set2', ax=ax)
plt.title("Visualizing Customer Clusters", fontsize=12, color="#ffbb39")
st.pyplot(fig)

# --- Prediction Form ---
st.markdown("<h3>🎯 Predict Your Shopping Personality</h3>", unsafe_allow_html=True)
with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.radio("Gender", ["Male", "Female"])
        age = st.slider("Age", 15, 70, 30)

    with col2:
        income = st.slider("Annual Income (k$)", 10, 150, 40)
        score = st.slider("Spending Score (1-100)", 1, 100, 50)

    submitted = st.form_submit_button("🧠 Predict My Segment")

# --- Prediction Result ---
if submitted:
    gender_encoded = 0 if gender == "Male" else 1
    user_input = [[gender_encoded, age, income, score]]
    user_cluster = kmeans.predict(user_input)[0]

    descriptions = {
        0: "🎁 Cluster 0: Budget hunters – love promotions and seasonal deals.",
        1: "👛 Cluster 1: High-income spenders – loyal to premium brands.",
        2: "🧢 Cluster 2: Trendy youth – stylish but budget-conscious.",
        3: "💎 Cluster 3: Mature loyalists – consistent big-ticket buyers.",
        4: "🛒 Cluster 4: Balanced buyers – stable income, stable spend."
    }

    st.markdown(f"<div class='custom-box'>{descriptions[user_cluster]}</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("<small style='color:gray;'>📘 Project for DAT 2102 – Information Security, Governance & the Cloud</small>", unsafe_allow_html=True)
st.markdown("""
<audio autoplay loop>
  <source src="https://www.bensound.com/bensound-music/bensound-happyrock.mp3" type="audio/mp3">
</audio>
""", unsafe_allow_html=True)

