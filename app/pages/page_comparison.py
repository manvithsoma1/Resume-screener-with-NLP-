import streamlit as st
import pandas as pd
import plotly.express as px
import os

def show():
    st.title("📈 Model Comparison & Evaluation")
    
    st.markdown("""
    This dashboard compares the performance metrics of the various classification models trained in the ML Pipeline.
    """)
    
    report_path = 'outputs/reports/model_comparison.csv'
    
    if os.path.exists(report_path):
        df = pd.read_csv(report_path)
        
        st.subheader("Performance Metrics Table")
        st.dataframe(df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(df, x='Model', y='Accuracy', color='Model', title="Model Accuracy Comparison")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            fig2 = px.bar(df, x='Model', y='F1-Score', color='Model', title="Model F1-Score Comparison")
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("### Efficiency Analysis")
        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.bar(df, x='Model', y='Train Time (s)', color='Model', title="Training Time (seconds)")
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            fig4 = px.bar(df, x='Model', y='Inference Time (s)', color='Model', title="Inference Time (seconds)")
            st.plotly_chart(fig4, use_container_width=True)
            
    else:
        st.warning("Model comparison data not found. Please run the backend training pipeline (`python src/train.py`).")

