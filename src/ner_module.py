import spacy
import pandas as pd
import re
import os

class NERExtractor:
    def __init__(self, skills_csv='data/skills/skills.csv'):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')
            
        self.skills_dict = self._load_skills(skills_csv)

    def _load_skills(self, skills_csv):
        if os.path.exists(skills_csv):
            df = pd.read_csv(skills_csv)
            # Create a lowercase set of skills for fast matching
            return set(df['skill'].str.lower().tolist())
        else:
            print(f"Warning: Skills dictionary not found at {skills_csv}")
            return set(['python', 'java', 'sql', 'machine learning', 'nlp', 'data science'])

    def extract_spacy_entities(self, text):
        """Extract entities using default spaCy model (ORG, GPE, PERSON, etc.)"""
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'explanation': spacy.explain(ent.label_)
            })
        return entities

    def extract_skills_dictionary(self, text):
        """Extract skills based on dictionary matching."""
        # Simple word tokenization and matching
        text_lower = text.lower()
        found_skills = set()
        
        # Check single words and bigrams from the dictionary
        for skill in self.skills_dict:
            # Word boundary regex to ensure exact match
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                found_skills.add(skill.title())
                
        return list(found_skills)

    def extract_all(self, text):
        """Combines spaCy NER and Custom Skill Extraction."""
        return {
            'entities': self.extract_spacy_entities(text),
            'skills': self.extract_skills_dictionary(text)
        }

if __name__ == "__main__":
    ner = NERExtractor()
    sample = "I worked at Google in New York. I am skilled in Python, Machine Learning, and SQL."
    print("NER Output:", ner.extract_all(sample))
