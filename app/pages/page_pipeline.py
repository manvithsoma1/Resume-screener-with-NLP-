import streamlit as st
import pandas as pd
from src.preprocessing import NLPPreprocessor
from src.parser_module import NLPParser
from src.ner_module import NERExtractor
from src.pdf_parser import PDFParser

def show():
    st.title("⚙️ Live NLP Pipeline")
    
    st.markdown("""
    This page demonstrates the step-by-step transformation of raw text into structured NLP features.
    Upload a resume or paste text to see the pipeline in action!
    """)
    
    # Initialize modules
    @st.cache_resource
    def load_modules():
        return NLPPreprocessor(), NLPParser(), NERExtractor(), PDFParser()
        
    preprocessor, parser, ner, pdf_parser = load_modules()
    
    upload_type = st.radio("Input Method:", ["Paste Text", "Upload PDF"])
    
    raw_text = ""
    if upload_type == "Paste Text":
        raw_text = st.text_area("Paste Resume Text Here:", height=200, value="John Doe\nSoftware Engineer at Google.\nSkills: Python, Machine Learning, AWS, SQL.\nI have 5 years of experience building scalable ML models.")
    else:
        uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")
        if uploaded_file is not None:
            raw_text = pdf_parser.parse_resume(uploaded_file)
            st.success("PDF Extracted Successfully!")
            with st.expander("Show Extracted Raw Text"):
                st.text(raw_text)

    if st.button("Run NLP Pipeline", type="primary") and raw_text:
        
        st.markdown("---")
        st.header("1. Text Cleaning & Normalization")
        st.info("Removes URLs, emails, special characters, and normalizes casing.")
        cleaned_text = preprocessor.clean_text(raw_text)
        st.code(cleaned_text, language="text")
        
        st.markdown("---")
        st.header("2. Tokenization")
        st.info("Splits text into individual words (tokens) for analysis.")
        tokens = preprocessor.tokenize_words(cleaned_text)
        st.write(tokens[:50]) # Show first 50
        
        st.markdown("---")
        st.header("3. Stopword Removal")
        st.info("Removes common words (e.g., 'the', 'is', 'in') that carry little semantic meaning.")
        no_stops = preprocessor.remove_stopwords(tokens)
        st.write(no_stops[:50])
        
        st.markdown("---")
        st.header("4. Stemming vs Lemmatization")
        st.info("Reduces words to their root form. Stemming is heuristic (fast, rough), Lemmatization uses dictionary context (slower, accurate).")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Stemming (Porter)")
            stemmed = preprocessor.stem_words(no_stops)
            st.write(stemmed[:30])
        with col2:
            st.subheader("Lemmatization (WordNet)")
            lemmatized = preprocessor.lemmatize_words(no_stops)
            st.write(lemmatized[:30])
            
        st.markdown("---")
        st.header("5. Part-of-Speech (POS) Tagging")
        st.info("Assigns grammatical categories (Noun, Verb, Adjective) to words.")
        pos_tags = parser.get_spacy_pos_tags(cleaned_text)
        
        # Display as a dataframe for neatness
        pos_df = pd.DataFrame(pos_tags, columns=['Word', 'POS', 'Tag', 'Explanation'])
        st.dataframe(pos_df.head(20), use_container_width=True)
        
        st.markdown("---")
        st.header("6. Named Entity Recognition (NER) & Skill Extraction")
        st.info("Identifies specific entities (Skills, Organizations, Locations) from the text.")
        
        ner_results = ner.extract_all(raw_text)
        
        col_skills, col_ents = st.columns(2)
        with col_skills:
            st.subheader("Extracted Skills")
            if ner_results['skills']:
                for skill in ner_results['skills']:
                    st.markdown(f"✅ **{skill}**")
            else:
                st.write("No specific skills found from dictionary.")
                
        with col_ents:
            st.subheader("Other Entities (spaCy)")
            if ner_results['entities']:
                ent_df = pd.DataFrame(ner_results['entities'])
                st.dataframe(ent_df, use_container_width=True)
            else:
                st.write("No entities found.")
