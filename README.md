# Intelligent Resume Screening, Candidate Ranking, and Semantic Job Matching System

This project is a complete, end-to-end NLP and Machine Learning system that automates the resume screening process. It uses classical NLP techniques, embeddings, and Machine Learning models to analyze, classify, and rank resumes against job descriptions.

## 🚀 Features
- **PDF Resume Parsing**: Upload and extract text directly from PDF resumes.
- **Classical NLP Pipeline**: Deeply educational steps demonstrating text cleaning, tokenization, stopword removal, stemming, lemmatization, POS tagging, and Dependency Parsing.
- **Named Entity Recognition (NER)**: Extracts skills, organizations, and custom entities using a hybrid dictionary and spaCy approach.
- **Feature Engineering & Embeddings**: Implements Bag of Words (BoW), TF-IDF, Word2Vec, and Sentence Transformers (`all-MiniLM-L6-v2`).
- **Machine Learning Classification**: Classifies resumes into industry categories using Naive Bayes, Logistic Regression, SVM, and Random Forest.
- **Semantic Job Matching**: Ranks candidates based on cosine similarity between job descriptions and resumes.
- **Explainable AI (XAI)**: Understand *why* a resume was selected by looking at feature importance and TF-IDF keywords.
- **Industry Metrics**: Includes Skill Gap Analysis, ATS Score calculation, and Top Missing Skills detection.
- **Modern Streamlit UI**: An interactive, 16-page educational UI to visualize the entire process.

## 📂 Project Structure
```text
project_root/
│
├── app/                  # Streamlit frontend pages and components
├── data/                 # Raw/processed datasets and skills dictionaries
├── models/               # Saved trained ML models and transformers
├── notebooks/            # Academic Jupyter Notebooks with theory
├── outputs/              # Generated plots, reports, and predictions
├── src/                  # Core modular backend Python scripts
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container deployment configuration
└── README.md             # Project documentation
```

## 🛠️ Tech Stack
- **Language**: Python 3.10
- **NLP**: NLTK, spaCy, Gensim, Sentence-Transformers, Transformers (Hugging Face)
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Plotly, Seaborn, WordCloud
- **UI**: Streamlit, Streamlit-Option-Menu
- **Utilities**: Pandas, NumPy, PyPDF2, pdfplumber

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone <repository_url>
cd nlp_project_resume_filter
```

### 2. Run Setup Script (Windows/Linux)
```bash
bash setup.sh
# or manually:
# pip install -r requirements.txt
# python -m spacy download en_core_web_sm
# python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### 3. Run the Streamlit Application
```bash
streamlit run app/main.py
```

## 🧠 NLP Concepts Covered
- Text Normalization & Preprocessing
- Lexical Analysis (Stemming vs Lemmatization)
- Syntactic Analysis (POS Tagging & Dependency Parsing)
- Semantic Analysis (Word/Sentence Embeddings)
- Information Retrieval (TF-IDF, Cosine Similarity)
- Named Entity Recognition (NER)
- Explainable AI in NLP

## 📊 Evaluation
The models are evaluated using Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and ROC Curves, with an interactive dashboard to compare their performances in real-time.
