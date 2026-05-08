import streamlit as st
import pandas as pd
from src.embeddings import EmbeddingGenerator
from src.similarity import SimilarityMatcher
from src.data_loader import load_resume_data
from src.job_description_generator import load_jds
from src.industry_metrics import IndustryMetrics
from src.ner_module import NERExtractor

def show():
    st.title("🏆 Candidate Ranking & Semantic Match")
    
    st.markdown("""
    This system ranks resumes against a Job Description using **Semantic Sentence Embeddings** (Sentence-Transformers) and **Cosine Similarity**. 
    It also provides Industry Metrics like **ATS Score** and **Skill Gap Analysis**.
    """)
    
    @st.cache_resource
    def load_engines():
        embedder = EmbeddingGenerator()
        matcher = SimilarityMatcher()
        metrics = IndustryMetrics()
        ner = NERExtractor()
        resumes_df = load_resume_data()
        # Keep a smaller subset for speed in demo
        resumes_df = resumes_df.sample(min(200, len(resumes_df)), random_state=42).reset_index(drop=True)
        # Pre-compute resume embeddings for speed
        with st.spinner("Initializing Resume Embeddings (Local Semantic Space)..."):
            res_texts = resumes_df['Resume'].tolist()
            res_embs = embedder.generate_embeddings(res_texts)
        return embedder, matcher, metrics, ner, resumes_df, res_embs
        
    embedder, matcher, metrics, ner, resumes_df, res_embs = load_engines()
    jds_df = load_jds()
    
    selected_jd_title = st.selectbox("Select a Job Description Template", jds_df['Job Title'].tolist())
    jd_text = jds_df[jds_df['Job Title'] == selected_jd_title]['Description'].values[0]
    
    st.text_area("Job Description Details", value=jd_text, height=150)
    
    if st.button("Rank Candidates", type="primary"):
        with st.spinner("Computing semantic similarity and ranking candidates..."):
            jd_emb = embedder.generate_embeddings([jd_text])[0]
            
            # Prepare metadata mapping
            metadata = resumes_df.to_dict('records')
            
            # Rank
            top_candidates = matcher.rank_items(jd_emb, res_embs, metadata, top_n=5)
            
            # Extract skills from JD for Gap Analysis
            jd_skills = ner.extract_skills_dictionary(jd_text)
            
            st.header("Top Ranked Resumes")
            for i, cand in enumerate(top_candidates):
                with st.expander(f"Rank {i+1} | Match: {cand['Similarity_Score']*100:.1f}% | Category: {cand['Category']}"):
                    st.progress(float(cand['Similarity_Score']))
                    
                    # Skill Gap Analysis
                    res_skills = ner.extract_skills_dictionary(cand['Resume'])
                    gap_analysis = metrics.skill_gap_analysis(res_skills, jd_skills)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("ATS Skill Match Score", f"{gap_analysis['Match_Percentage']}%")
                    with col2:
                        st.markdown("**Matched Skills**")
                        st.write(", ".join(gap_analysis['Matched_Skills']) if gap_analysis['Matched_Skills'] else "None")
                    with col3:
                        st.markdown("**Missing Skills**")
                        st.write(", ".join(gap_analysis['Missing_Skills']) if gap_analysis['Missing_Skills'] else "None")
                        
                    st.markdown("**Resume Snippet:**")
                    st.text(cand['Resume'][:500] + "...")
