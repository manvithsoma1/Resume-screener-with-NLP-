import streamlit as st
import pickle
import os
import pandas as pd
from src.preprocessing import NLPPreprocessor
from src.pdf_parser import PDFParser
import plotly.express as px

def show():
    st.title("🤖 Resume Classification")
    
    st.markdown("""
    This page uses our trained Machine Learning models (based on TF-IDF features) to classify a resume into a specific industry category.
    """)
    
    @st.cache_resource
    def load_artifacts():
        preprocessor = NLPPreprocessor()
        pdf_parser = PDFParser()
        
        models = {}
        vectorizer = None
        label_encoder = None
        
        try:
            with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            with open('models/label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
                
            model_files = [f for f in os.listdir('models') if f.endswith('_model.pkl')]
            for mf in model_files:
                name = mf.replace('_model.pkl', '').replace('_', ' ').title()
                with open(os.path.join('models', mf), 'rb') as f:
                    models[name] = pickle.load(f)
        except Exception as e:
            st.error("Models not found. Please run backend training first.")
            
        return preprocessor, pdf_parser, vectorizer, label_encoder, models
        
    preprocessor, pdf_parser, vectorizer, label_encoder, models = load_artifacts()
    
    if not models:
        return
        
    selected_model_name = st.selectbox("Select Classification Model", list(models.keys()))
    selected_model = models[selected_model_name]
    
    upload_type = st.radio("Input Method:", ["Paste Text", "Upload PDF"], horizontal=True)
    
    raw_text = ""
    if upload_type == "Paste Text":
        raw_text = st.text_area("Paste Resume Text Here:", height=200, value="Experienced Data Scientist with 4 years in Python, Machine Learning, and Deep Learning.")
    else:
        uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")
        if uploaded_file is not None:
            raw_text = pdf_parser.parse_resume(uploaded_file)
            st.success("PDF Extracted")

    if st.button("Predict Category", type="primary") and raw_text:
        with st.spinner("Processing..."):
            cleaned = preprocessor.full_pipeline(raw_text)
            features = vectorizer.transform([cleaned])
            
            prediction = selected_model.predict(features)[0]
            predicted_class = label_encoder.inverse_transform([prediction])[0]
            
            st.success(f"### Predicted Category: **{predicted_class}**")
            
            if hasattr(selected_model, "predict_proba"):
                probs = selected_model.predict_proba(features)[0]
                prob_df = pd.DataFrame({
                    'Category': label_encoder.classes_,
                    'Probability': probs
                })
                prob_df = prob_df.sort_values(by='Probability', ascending=False).head(5)
                
                fig = px.bar(prob_df, x='Probability', y='Category', orientation='h', 
                             title=f"Top 5 Predictions (Confidence) - {selected_model_name}",
                             color='Probability', color_continuous_scale='viridis')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("This model does not support probability estimates.")
