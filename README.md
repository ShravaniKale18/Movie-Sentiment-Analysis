🎬 IMDB Movie Review Sentiment AnalyzerA Machine Learning-powered Web Application built with Streamlit, Scikit-Learn, and NLTK that classifies movie reviews as either Positive or Negative with high accuracy and confidence scoring.📌 FeaturesText Preprocessing: Automated removal of HTML tags, punctuation, and English stopwords.Bag-of-Words Vectorization: Utilizes a fitted CountVectorizer with up to 5,000 top features.High Performance: Powered by a tuned Logistic Regression model trained on the IMDB dataset.Interactive UI: Real-time sentiment prediction with confidence probability scores built using Streamlit.Robust NLTK Handling: Efficient and reliable local downloading of NLTK resources (stopwords) without infinite execution loops.📊 Model Training & EvaluationThe classification models were trained and benchmarked on the IMDB Dataset (50,000 reviews) after deduplication (49,582 unique records).Model BenchmarksAlgorithmAccuracy ScoreStatusGaussian Naive Bayes (GNB)~76.42%EvaluatedMultinomial Naive Bayes (MNB)~84.56%EvaluatedLogistic Regression (LR)~87.04%Selected & Exported📂 Project StructureMovie Review Classification/
│
├── app.py                   # Main Streamlit web application script
├── best_model_lr.pkl        # Serialized Logistic Regression model
├── vectorizer.pkl           # Serialized CountVectorizer instance
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── nltk_data/               # Local cache for downloaded NLTK datasets
⚙️ Installation & Setup1. Clone or Open the RepositoryNavigate to your project folder in your terminal:cd "Movie Review Classification"
2. Set Up a Virtual Environment (Recommended)python -m venv .venv
source .venv/bin/activate   # On Linux/macOS
# .venv\Scripts\activate    # On Windows
3. Install Required DependenciesEnsure you have all necessary packages installed by running:pip install -r requirements.txt
Sample requirements.txt:streamlit
scikit-learn
pandas
numpy
nltk
🚀 Running the Web ApplicationLaunch the Streamlit app using:streamlit run app.py
Open your browser at http://localhost:8501 to view and test the application.🧪 Usage ExampleOpen the application interface.Enter a movie review in the text box (e.g., "The visual effects were stunning and the lead acting kept me hooked throughout!").Click Analyze Sentiment.View the result detailing whether the review is Positive 🔥 or Negative ❄️ along with the probability confidence level.
An updated `README.md` file has been created. It covers your project architecture, training benchmarks from your Colab notebook (Logistic Regression ~87.04% accuracy), installation steps, and instructions for launching the Streamlit app.
