#!/bin/bash
# Setup script for the NLP project

echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

echo "Setup complete! Run 'streamlit run app/main.py' to start the application."
