# 🎬 IMDB Movie Review Sentiment Analyzer

A Machine Learning-powered Web Application built with **Streamlit**, **Scikit-Learn**, and **NLTK** that classifies movie reviews as either **Positive** or **Negative** with high accuracy and confidence scoring.

---

## 📌 Features

- **Text Preprocessing**: Automated removal of HTML tags, punctuation, and English stopwords.
- **Bag-of-Words Vectorization**: Utilizes a fitted `CountVectorizer` with up to 5,000 top features.
- **High Performance**: Powered by a tuned **Logistic Regression** model trained on the IMDB dataset.
- **Interactive UI**: Real-time sentiment prediction with confidence probability scores built using Streamlit.
- **Robust NLTK Handling**: Efficient and reliable local downloading of NLTK resources (`stopwords`) without infinite execution loops.

---

## 📊 Model Training & Evaluation

The classification models were trained and benchmarked on the **IMDB Dataset** (50,000 reviews) after deduplication (49,582 unique records).

### Model Benchmarks
| Algorithm | Accuracy Score | Status |
| :--- | :---: | :---: |
| **Gaussian Naive Bayes (GNB)** | ~76.42% | Evaluated |
| **Multinomial Naive Bayes (MNB)** | ~84.56% | Evaluated |
| **Logistic Regression (LR)** | **~87.04%** | **Selected & Exported** |

---

## 📂 Project Structure

```text
Movie Review Classification/
│
├── app.py                   # Main Streamlit web application script
├── best_model_lr.pkl        # Serialized Logistic Regression model
├── vectorizer.pkl           # Serialized CountVectorizer instance
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── nltk_data/               # Local cache for downloaded NLTK datasets
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open the Repository
Navigate to your project folder in your terminal:
```bash
cd "Movie Review Classification"
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # On Linux/macOS
# .venv\Scripts\activate    # On Windows
```

### 3. Install Required Dependencies
Ensure you have all necessary packages installed by running:
```bash
pip install -r requirements.txt
```

*Sample `requirements.txt`:*
```text
streamlit
scikit-learn
pandas
numpy
nltk
```

---

## 🚀 Running the Web Application

Launch the Streamlit app using:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to view and test the application.

---

## 🧪 Usage Example

1. Open the application interface.
2. Enter a movie review in the text box (e.g., *"The visual effects were stunning and the lead acting kept me hooked throughout!"*).
3. Click **Analyze Sentiment**.
4. View the result detailing whether the review is **Positive** 🔥 or **Negative** ❄️ along with the probability confidence level.
```

An updated `README.md` file has been created. It covers your project architecture, training benchmarks from your Colab notebook (Logistic Regression ~87.04% accuracy), installation steps, and instructions for launching the Streamlit app.