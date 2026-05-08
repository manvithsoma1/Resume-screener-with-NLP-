import pandas as pd
import numpy as np
import os
import random

def generate_synthetic_resumes(num_samples=200):
    """Generates a synthetic resume dataset if the real one is missing."""
    categories = ['Data Science', 'HR', 'Java Developer', 'Python Developer', 
                  'Testing', 'Web Designing', 'DevOps', 'Business Analyst', 
                  'Mechanical Engineer', 'NLP Engineer']
    
    data = []
    
    skills_pool = {
        'Data Science': ['Python', 'Machine Learning', 'Data Science', 'Pandas', 'NumPy', 'Scikit-learn', 'TensorFlow', 'Keras', 'SQL', 'Data Analysis', 'Tableau', 'Statistics'],
        'HR': ['Recruitment', 'Human Resources', 'Sourcing', 'Employee Relations', 'Onboarding', 'Talent Acquisition', 'HR Policies', 'Communication', 'Payroll', 'Interviewing'],
        'Java Developer': ['Java', 'Spring Boot', 'Hibernate', 'J2EE', 'SQL', 'Maven', 'Tomcat', 'REST API', 'Microservices', 'Git', 'Agile', 'Junit'],
        'Python Developer': ['Python', 'Django', 'Flask', 'FastAPI', 'REST API', 'PostgreSQL', 'Docker', 'AWS', 'Linux', 'Git', 'Unit Testing', 'Redis'],
        'Testing': ['Manual Testing', 'Automation Testing', 'Selenium', 'Java', 'Python', 'Jira', 'TestNG', 'Cucumber', 'API Testing', 'Postman', 'Agile', 'Bug Tracking'],
        'Web Designing': ['HTML', 'CSS', 'JavaScript', 'React', 'Angular', 'UI/UX', 'Figma', 'Adobe XD', 'Bootstrap', 'Responsive Design', 'Web Development'],
        'DevOps': ['Docker', 'Kubernetes', 'AWS', 'Jenkins', 'CI/CD', 'Linux', 'Bash', 'Terraform', 'Ansible', 'Git', 'Monitoring', 'Prometheus'],
        'Business Analyst': ['Business Analysis', 'Requirement Gathering', 'Agile', 'Scrum', 'SQL', 'Data Analysis', 'Jira', 'UML', 'Communication', 'Stakeholder Management', 'Excel', 'PowerBI'],
        'Mechanical Engineer': ['AutoCAD', 'SolidWorks', 'Mechanical Engineering', 'CAD', 'Manufacturing', 'Design', 'ANSYS', 'Product Design', 'Engineering', 'Matlab', 'Project Management'],
        'NLP Engineer': ['Python', 'NLP', 'Natural Language Processing', 'spaCy', 'NLTK', 'Transformers', 'BERT', 'Deep Learning', 'PyTorch', 'Text Classification', 'Word2Vec', 'LLMs']
    }
    
    intro_pool = [
        "Experienced professional with a proven track record in {}. ",
        "Dedicated {} specialist with strong background. ",
        "Results-oriented {} looking for challenging opportunities. ",
        "Passionate about {} and building scalable solutions. ",
        "Highly skilled {} with over 5 years of experience. "
    ]
    
    edu_pool = [
        "Bachelor's degree in Computer Science. ",
        "Master's degree in Engineering. ",
        "B.Tech in Information Technology. ",
        "MBA in Human Resources. ",
        "B.S. in relevant field. "
    ]
    
    for _ in range(num_samples):
        cat = random.choice(categories)
        
        # Build text
        intro = random.choice(intro_pool).format(cat)
        edu = random.choice(edu_pool)
        
        # Pick 5-8 skills
        num_skills = random.randint(5, 8)
        skills = random.sample(skills_pool[cat], min(num_skills, len(skills_pool[cat])))
        skills_text = "Skills: " + ", ".join(skills) + ". "
        
        # Add some random other words to simulate real text
        exp = f"Worked on various projects involving {skills[0]} and {skills[1]}. Implemented solutions that improved efficiency by 20%. "
        
        resume_text = intro + edu + exp + skills_text
        
        data.append({
            'Category': cat,
            'Resume': resume_text
        })
        
    df = pd.DataFrame(data)
    return df

def load_resume_data(file_path='data/raw/Resume.csv'):
    """Loads resume data from CSV, or generates synthetic data if missing."""
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        if 'Resume_str' in df.columns:
             df = df.rename(columns={'Resume_str': 'Resume'})
        if 'Category' not in df.columns and 'Category' in df.columns.str.title():
             df.columns = df.columns.str.title()
        return df[['Category', 'Resume']].dropna()
    else:
        print(f"Warning: {file_path} not found. Auto-generating synthetic dataset.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df = generate_synthetic_resumes(num_samples=500)
        df.to_csv(file_path, index=False)
        return df

if __name__ == "__main__":
    df = load_resume_data()
    print(f"Dataset shape: {df.shape}")
    print(df['Category'].value_counts())
