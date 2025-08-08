import streamlit as st
import joblib
import nltk
import string
import numpy as np
import random
from nltk.corpus import stopwords
import requests
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

nltk.download('stopwords')

# Load model and vectorizer
try:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    st.error(f"❌ Error loading model/vectorizer: {e}")

# -----------------------------
# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# -----------------------------
# Example news samples
examples = {
    "🔴 Fake: Earth is flat": "Scientists confirm Earth is flat and NASA has been lying for decades.",
    "🔴 Fake: Banana peels cure cancer": "Cure for cancer found in banana peels, say anonymous doctors.",
    "🗾 Real: NASA launches Artemis I": "NASA successfully launched the Artemis I mission from the Kennedy Space Center.",
    "🗾 Real: Govt mental health funding": "The government has announced new funding for mental health programs nationwide.",
    "✍️ Enter your own": ""
}

# -----------------------------
def fetch_similar_real_news(user_text):
    api_key = "11b036c7d4c54d5f924060c93f1facf4"

    sentences = re.split(r'[\n.;!?]', user_text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) > 4]

    query = sentences[0] if sentences else user_text
    query = re.sub(r"[^a-zA-Z0-9\s]", "", query)
    query = query[:100]

    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&pageSize=3&apiKey={api_key}"

    try:
        response = requests.get(url)
        data = response.json()

        if data["status"] != "ok" or not data["articles"]:
            return []

        headlines = [
            article["title"] + " — " + article.get("description", "")
            for article in data["articles"]
            if article.get("title")
        ]
        return headlines

    except Exception as e:
        st.warning(f"🌐 Error fetching live news: {e}")
        return []

# -----------------------------
from sklearn.metrics.pairwise import cosine_similarity

def get_cosine_similarity_score(text1, text_list):
    texts = [text1] + text_list
    tfidf = vectorizer.transform(texts)
    similarity_matrix = cosine_similarity(tfidf[0:1], tfidf[1:])
    return float(np.max(similarity_matrix))  # Use max similarity value

# -----------------------------
st.sidebar.header("🔍 Optional Filters")
filter_source = st.sidebar.text_input("News Source (e.g., BBC, Reuters)")
filter_date = st.sidebar.text_input("Date (e.g., 2021)")

# -----------------------------
st.set_page_config(
    page_title="Fake News Classifier",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    body {
        background-color: rgb(250, 250, 250);
    }
    .stProgress > div > div > div > div {
        background-color: #ff4b4b;
    }
    .example-box {
        background-color: #f1f3f6;
        padding: 0.75em;
        border-radius: 10px;
        font-size: 0.9em;
        margin-bottom: 1em;
    }
</style>
""", unsafe_allow_html=True)

st.image("logo.png", width=100)
st.title("📰 Fake News Classifier with Smart Confidence & Real News Similarity")
st.caption("A simple, elegant tool to detect **Fake News** using Machine Learning")
st.markdown("---")

# -----------------------------
with st.sidebar:
    st.image("logo.png", width=120)
    st.header("ℹ️ About This App")
    st.write("""
    This app uses a **Logistic Regression** model trained on real & fake news to classify articles.

    **🔧 Technologies Used:**
    - **Model:** Logistic Regression  
    - **Vectorizer:** TF-IDF  
    - **Dataset:** [Fake & Real News (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

    💡 Developed by [Shrija Tewari](https://github.com/shrijatewari)
    """)
    st.markdown("🔗 [GitHub Repository](https://github.com/shrijatewari)")

# -----------------------------
st.subheader("📝 Enter News Article")
selected_example = st.selectbox("🧪 Try a sample or write your own:", list(examples.keys()))
def_text = examples[selected_example]

col1, col2 = st.columns(2)
with col1:
    if st.button("🎲 Try Random Real Example"):
        def_text = random.choice([v for k, v in examples.items() if k.startswith("🗾")])
with col2:
    if st.button("🔝 Try Random Fake Example"):
        def_text = random.choice([v for k, v in examples.items() if k.startswith("🔴")])

user_input = st.text_area("Paste or type the article here:", value=def_text, height=250)

# -----------------------------
if user_input and st.button("🔍Predict"):
    with st.spinner("Analyzing the article..."):
        try:
            # Step 1: Predict using model
            input_vector = vectorizer.transform([user_input])  # Ensure 2D shape
            prediction = model.predict(input_vector)[0]
            original_confidence = model.predict_proba(input_vector)[0][1]

            # Step 2: Fetch real news
            real_news = fetch_similar_real_news(user_input)

            similarity_score = 0
            similarity_boost = 0

            # Step 3: Compute similarity boost
            if real_news:
                try:
                    similarity_score = get_cosine_similarity_score(user_input, real_news)
                    if similarity_score > 0.3:
                        similarity_boost = similarity_score * 0.5
                    else:
                        similarity_boost = 0
                except Exception as e:
                    st.warning(f"Similarity scoring failed: {e}")
                    similarity_boost = 0
            else:
                st.info("🔍 No matching real news found online.")

            # Step 4: Add boost to confidence
            boosted_confidence = original_confidence + similarity_boost
            boosted_confidence = min(boosted_confidence, 1.0)

            # Step 5: Final prediction logic
            final_prediction = 'FAKE' if boosted_confidence < 0.5 else 'REAL'

            # Step 6: Display result
            if final_prediction == 'FAKE':
                st.error("🚨 This article is predicted to be FAKE NEWS.")
            else:
                st.success("✅ This article is predicted to be REAL NEWS.")

            st.metric(label="Confidence Score", value=f"{boosted_confidence * 100:.2f}%")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
