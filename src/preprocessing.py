import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy

# Ensure NLTK resources are available (usually handled by setup.sh)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    pass

class NLPPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')

    def clean_text(self, text):
        """Removes noise, URLs, emails, special characters."""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_words(self, text):
        return word_tokenize(text)

    def tokenize_sentences(self, text):
        return sent_tokenize(text)

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def stem_words(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]

    def lemmatize_words(self, tokens):
        return [self.lemmatizer.lemmatize(word) for word in tokens]
        
    def spacy_lemmatize(self, text):
        doc = self.nlp(text)
        return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]

    def full_pipeline(self, text, use_spacy=False):
        """Runs the entire preprocessing pipeline and returns a cleaned string."""
        cleaned = self.clean_text(text)
        if use_spacy:
            lemmatized = self.spacy_lemmatize(cleaned)
            return " ".join(lemmatized)
        else:
            tokens = self.tokenize_words(cleaned)
            no_stops = self.remove_stopwords(tokens)
            lemmatized = self.lemmatize_words(no_stops)
            return " ".join(lemmatized)

if __name__ == "__main__":
    preprocessor = NLPPreprocessor()
    sample = "I am a Data Scientist! I love building NLP models in Python. Contact me at john@email.com or visit https://john.com"
    print("Original:", sample)
    print("Cleaned:", preprocessor.clean_text(sample))
    tokens = preprocessor.tokenize_words(preprocessor.clean_text(sample))
    print("Tokens:", tokens)
    print("No Stops:", preprocessor.remove_stopwords(tokens))
    print("Lemmatized:", preprocessor.lemmatize_words(preprocessor.remove_stopwords(tokens)))
    print("Full Pipeline:", preprocessor.full_pipeline(sample))
