import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_resume_data

def show():
    st.title("📊 Dataset Explorer")
    
    st.markdown("""
    > [!NOTE]
    > **What is this?** This page explores the dataset used to train the machine learning models. High-quality data is the foundation of any robust NLP system.
    """)
    
    @st.cache_data
    def load_data():
        return load_resume_data()
        
    try:
        df = load_data()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Resumes", len(df))
        with col2:
            st.metric("Total Categories", df['Category'].nunique())
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
            
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Category Distribution")
        category_counts = df['Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        fig = px.bar(category_counts, x='Category', y='Count', 
                     color='Count', color_continuous_scale='Blues',
                     title='Number of Resumes per Category')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.warning("Please run the backend training scripts to generate the synthetic datasets if the Kaggle dataset is not present.")
