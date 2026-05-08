import streamlit as st
import pickle
import os
import pandas as pd
import plotly.express as px
from src.preprocessing import NLPPreprocessor
from src.pdf_parser import PDFParser
from src.explainability import ModelExplainer

def show():
    st.title("🔍 Explainable AI (XAI) - Why This Resume?")
    
    st.markdown("""
    > [!IMPORTANT]
    > **Model Transparency**
    > This dashboard explains *how* the model reached its decision by revealing the internal TF-IDF weights and Model Coefficients (Feature Importances). 
    """)
    
    @st.cache_resource
    def load_explainer():
        preprocessor = NLPPreprocessor()
        pdf_parser = PDFParser()
        
        try:
            with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            with open('models/label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
            with open('models/logistic_regression_model.pkl', 'rb') as f:
                model = pickle.load(f) # Defaulting to LR for explainability
                
            explainer = ModelExplainer(model, vectorizer, label_encoder)
            return preprocessor, pdf_parser, explainer
        except Exception as e:
            st.error("Missing models. Please run training first.")
            return None, None, None

    preprocessor, pdf_parser, explainer = load_explainer()
    if not explainer: return
    
    upload_type = st.radio("Input Method:", ["Paste Text", "Upload PDF"], horizontal=True)
    raw_text = ""
    if upload_type == "Paste Text":
        raw_text = st.text_area("Paste Resume Text Here:", height=200, value="Strong Java developer with Spring Boot and Hibernate experience.")
    else:
        uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")
        if uploaded_file is not None:
            raw_text = pdf_parser.parse_resume(uploaded_file)
            st.success("PDF Extracted")

    if st.button("Explain Decision", type="primary") and raw_text:
        with st.spinner("Analyzing feature weights..."):
            explanation = explainer.explain_prediction(raw_text, preprocessor, top_n=15)
            
            st.success(f"### Predicted Category: **{explanation['Predicted_Class']}**")
            
            if explanation['Confidence']:
                st.metric("Model Confidence", f"{explanation['Confidence']*100:.2f}%")
                
            st.markdown("### Top Keywords Driving this Prediction")
            st.info("The chart below shows the words in the resume that had the highest positive influence on predicting this specific category.")
            
            # Unpack the top features
            # feature_importance.append((word, tfidf_score, coef, importance))
            df_features = pd.DataFrame(explanation['Top_Features'], columns=['Keyword', 'TF-IDF_Score', 'Model_Weight', 'Overall_Importance'])
            df_features = df_features.sort_values(by='Overall_Importance', ascending=True) # Ascending for horizontal bar chart
            
            fig = px.bar(df_features, x='Overall_Importance', y='Keyword', orientation='h',
                         hover_data=['TF-IDF_Score', 'Model_Weight'],
                         title="Feature Importance (TF-IDF * Model Weight)",
                         color='Overall_Importance', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Show Data Table"):
                st.dataframe(df_features.sort_values(by='Overall_Importance', ascending=False), use_container_width=True)
