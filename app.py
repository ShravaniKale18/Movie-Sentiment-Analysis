import os
import re
import string
import pickle
import streamlit as st
import nltk

nltk_data_path = os.path.abspath('./nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)

if nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_path)

# Download stopwords quietly if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
punct_table = str.maketrans('', '', string.punctuation)
html_cleaner = re.compile(r'<.*?>')

# 2. Optimized Preprocessing Helper Functions

def clean_review(text: str) -> str:
    """Cleans input text: lowercasing, HTML stripping, punctuation removal, and stopword removal."""
    text = text.lower()
    text = html_cleaner.sub('', text)
    text = text.translate(punct_table)
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    return ' '.join(filtered_words)

# 3. Asset Loading

@st.cache_resource
def load_assets():
    with open('vectorizer.pkl', 'rb') as cv_file:
        cv = pickle.load(cv_file)
    with open('best_model_lr.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return cv, model

# 4. Streamlit UI

st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to analyze whether the sentiment is positive or negative.")

# Load models and check for errors

try:
    cv, model = load_assets()
except Exception as e:
    st.error(f"⚠️ Error loading classification models: {e}")
    st.stop()

# User input text area
user_input = st.text_area(
    "Write your review here:", 
    height=150, 
    placeholder="I absolutely loved this movie! The acting was superb and the story kept me hooked..."
)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please write some text before analyzing!")
    else:
        # Preprocess text
        cleaned_text = clean_review(user_input)

        # Vectorize input text
        input_bow = cv.transform([cleaned_text])

        # Model predictions
        prediction = model.predict(input_bow)[0]
        prediction_proba = model.predict_proba(input_bow)[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]

        st.subheader("Analysis Result:")
        if sentiment == "Positive":
            st.success(f"🔥 **{sentiment}** Sentiment (Confidence: {confidence:.2%})")
        else:
            st.error(f"❄️ **{sentiment}** Sentiment (Confidence: {confidence:.2%})")