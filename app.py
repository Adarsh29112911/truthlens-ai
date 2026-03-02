import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="📰",
    layout="wide"
)


model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.title-text {
    font-size: 48px;
    font-weight: 700;
    text-align: center;
}
.subtitle-text {
    font-size: 22px;
    text-align: center;
    color: #A9A9A9;
}
.section-title {
    font-size: 28px;
    font-weight: 600;
}
.feature-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #1c1f26;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📌 Input News Article")
user_input = st.sidebar.text_area("Paste news article below:")

analyze_button = st.sidebar.button("🔎 Analyze News")

st.sidebar.markdown("---")
st.sidebar.info(
    "This system uses Machine Learning to classify news as Real or Fake "
    "based on textual patterns."
)

st.markdown('<p class="title-text">📰 TruthLens AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle-text">AI-Powered Fake News Detection System</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="text-align:center;">Built by Team AAR Intelligence<br>'
    'Adarsh Pandey • Abhishek • Raza</p>',
    unsafe_allow_html=True
)

st.markdown("---")


st.markdown('<p class="section-title">🚀 Why TruthLens AI?</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
    <h4>🔍 Smart Text Analysis</h4>
    Uses NLP & TF-IDF vectorization  
    to deeply analyze linguistic patterns.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
    <h4>🤖 Machine Learning Model</h4>
    Trained classification model  
    to detect fake vs real news.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
    <h4>⚡ Instant Prediction</h4>
    Get real-time results  
    within seconds.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown('<p class="section-title">🧠 How It Works</p>', unsafe_allow_html=True)

st.markdown("""
1️⃣ User pastes a news article  
2️⃣ Text is cleaned & vectorized  
3️⃣ ML model predicts authenticity  
4️⃣ Result displayed with confidence score  
""")

st.markdown("---")

st.markdown('<p class="section-title">📊 Model Performance</p>', unsafe_allow_html=True)

m1, m2 = st.columns(2)

with m1:
    st.metric("Model Accuracy", "94%")

with m2:
    st.metric("Model Type", "Logistic Regression")

st.markdown("---")

if analyze_button:
    if user_input.strip() == "":
        st.warning("⚠ Please enter a news article.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        probability = model.predict_proba(transformed_input).max() * 100

        st.markdown("## 🔎 Analysis Result")

        if prediction == 1:
            st.success(f"✅ This news appears to be REAL")
        else:
            st.error(f"❌ This news appears to be FAKE")

        st.info(f"Confidence Score: {probability:.2f}%")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2026 TruthLens AI | "
    "Hackathon Project | Powered by Machine Learning</p>",
    unsafe_allow_html=True
)