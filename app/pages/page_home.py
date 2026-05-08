import streamlit as st

def show():
    st.title("Intelligent Resume Screening & Ranking System")
    st.markdown("### A Complete NLP & Machine Learning Pipeline")
    
    st.markdown("""
    Welcome to the AI Resume Screener. This system is designed to automate the recruitment process using advanced Natural Language Processing (NLP) and Machine Learning techniques.
    
    ### 🎯 Project Objectives
    1. **Preprocess and Clean**: Convert raw text/PDFs into structured NLP formats.
    2. **Analyze**: Use Lexical and Syntactic analysis (POS, Dependency Parsing).
    3. **Extract**: Identify key skills using Named Entity Recognition (NER).
    4. **Classify**: Predict the industry category of the resume using trained ML models.
    5. **Rank**: Match resumes against Job Descriptions using semantic similarity.
    6. **Explain**: Provide transparent reasoning (Explainable AI) for model decisions.
    
    ### 🏗️ Architecture
    """)
    
    # We use a markdown flowchart to represent architecture visually
    st.markdown("""
    ```mermaid
    graph TD
        A[Raw Resume PDF/Text] --> B(NLP Preprocessing)
        B --> C{Feature Extraction}
        C -->|TF-IDF| D[Classification Models]
        C -->|Embeddings| E[Semantic Search]
        D --> F[Category Prediction]
        E --> G[Resume Ranking vs JD]
        F --> H((Explainable AI))
        G --> H
    ```
    """)
    
    st.info("👈 Use the **Sidebar Navigation** to explore each step of the pipeline interactively!")
    
    st.markdown("### 📚 Educational Value")
    st.markdown("""
    This platform serves as an interactive learning tool. On every page, you will find explanations detailing:
    - **What** the NLP concept is.
    - **Why** it is used in industry.
    - **How** the mathematical or programmatic implementation works.
    """)
