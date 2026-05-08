import spacy
from nltk import pos_tag, word_tokenize

class NLPParser:
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')

    def get_nltk_pos_tags(self, text):
        """Returns POS tags using NLTK."""
        tokens = word_tokenize(text)
        return pos_tag(tokens)

    def get_spacy_pos_tags(self, text):
        """Returns POS tags using spaCy."""
        doc = self.nlp(text)
        return [(token.text, token.pos_, token.tag_, spacy.explain(token.tag_)) for token in doc]

    def get_dependency_parse(self, text):
        """Returns dependency parsing tree using spaCy."""
        doc = self.nlp(text)
        return [(token.text, token.dep_, token.head.text, spacy.explain(token.dep_)) for token in doc]

    def extract_noun_chunks(self, text):
        """Extracts noun phrases which are often useful for skill detection."""
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]

if __name__ == "__main__":
    parser = NLPParser()
    sample = "The quick brown fox jumps over the lazy dog."
    print("NLTK POS:", parser.get_nltk_pos_tags(sample))
    print("spaCy POS:", parser.get_spacy_pos_tags(sample))
    print("Dependency:", parser.get_dependency_parse(sample))
