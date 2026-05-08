import pandas as pd
import os

def generate_job_descriptions(output_path='data/job_descriptions/jds.csv'):
    """Generates a dataset of synthetic job descriptions."""
    
    jds = [
        {
            "Job Title": "Data Scientist",
            "Category": "Data Science",
            "Description": "We are looking for a Data Scientist to analyze large amounts of raw information to find patterns that will help improve our company. We will rely on you to build data products to extract valuable business insights. Skills: Python, R, SQL, Machine Learning, Deep Learning, Pandas, NumPy, Scikit-learn, TensorFlow. Excellent analytical skills and problem-solving ability."
        },
        {
            "Job Title": "NLP Engineer",
            "Category": "NLP Engineer",
            "Description": "We are seeking an NLP Engineer to join our AI team. You will be responsible for transforming natural language data into useful features using NLP techniques. Skills required: Python, Natural Language Processing, spaCy, NLTK, Transformers, BERT, PyTorch, Word2Vec, LLMs. Experience with text classification and sentiment analysis."
        },
        {
            "Job Title": "Java Backend Developer",
            "Category": "Java Developer",
            "Description": "Hiring a Java Developer with experience in building high-performing, scalable, enterprise-grade applications. You will be responsible for Java/Java EE application development while providing expertise in the full software development lifecycle. Skills: Java, Spring Boot, Hibernate, SQL, Microservices, REST API, Git."
        },
        {
            "Job Title": "Frontend React Developer",
            "Category": "Web Designing",
            "Description": "Looking for a skilled Web Designer / Frontend Developer to design and implement attractive web interfaces. You should have a strong understanding of UI/UX principles. Skills: HTML, CSS, JavaScript, React, Angular, Figma, Responsive Design."
        },
        {
            "Job Title": "DevOps Engineer",
            "Category": "DevOps",
            "Description": "We are hiring a DevOps Engineer to help us build functional systems that improve customer experience. Responsibilities include deploying product updates, identifying production issues and implementing integrations. Skills: Docker, Kubernetes, AWS, Jenkins, CI/CD, Linux, Terraform."
        },
        {
            "Job Title": "Senior Python Developer",
            "Category": "Python Developer",
            "Description": "We are looking for a Python Developer to join our engineering team and help us develop and maintain various software products. Skills: Python, Django, Flask, FastAPI, PostgreSQL, AWS, Docker. Strong understanding of REST APIs and system design."
        },
        {
            "Job Title": "Quality Assurance Automation Tester",
            "Category": "Testing",
            "Description": "Seeking a QA Automation Tester to design testing procedures for our software applications. You will be responsible for analyzing functionality and reporting bugs. Skills: Automation Testing, Selenium, Java, Python, Jira, API Testing, Postman."
        },
        {
            "Job Title": "HR Manager",
            "Category": "HR",
            "Description": "We need an HR Manager to oversee all aspects of human resources practices and processes. You will support business needs and ensure the proper implementation of company strategy and objectives. Skills: Recruitment, Human Resources, Talent Acquisition, Employee Relations, Communication."
        },
        {
            "Job Title": "Business Analyst",
            "Category": "Business Analyst",
            "Description": "Looking for a Business Analyst to enhance our business processes. You will evaluate our current systems, discover areas for improvement, and implement solutions. Skills: Business Analysis, Agile, Scrum, Requirement Gathering, SQL, Data Analysis."
        },
        {
            "Job Title": "Mechanical Design Engineer",
            "Category": "Mechanical Engineer",
            "Description": "Hiring a Mechanical Engineer to design and develop mechanical systems. You will be involved in the full engineering lifecycle, from concept to production. Skills: AutoCAD, SolidWorks, Mechanical Engineering, CAD, ANSYS, Product Design."
        }
    ]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(jds)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} job descriptions at {output_path}")
    return df

def load_jds(file_path='data/job_descriptions/jds.csv'):
    if not os.path.exists(file_path):
        return generate_job_descriptions(file_path)
    return pd.read_csv(file_path)

if __name__ == "__main__":
    generate_job_descriptions()
