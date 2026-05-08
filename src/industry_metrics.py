class IndustryMetrics:
    def __init__(self):
        pass

    def calculate_ats_score(self, resume_skills, jd_skills):
        """Calculates a simulated ATS (Applicant Tracking System) Score."""
        if not jd_skills:
            return 0.0
            
        # Convert to lowercase sets for comparison
        res_skills_set = set([s.lower() for s in resume_skills])
        jd_skills_set = set([s.lower() for s in jd_skills])
        
        matches = len(res_skills_set.intersection(jd_skills_set))
        total_required = len(jd_skills_set)
        
        if total_required == 0:
            return 0.0
            
        score = (matches / total_required) * 100
        return round(score, 2)

    def skill_gap_analysis(self, resume_skills, jd_skills):
        """Identifies matched and missing skills between Resume and Job Description."""
        res_skills_lower = {s.lower(): s for s in resume_skills}
        jd_skills_lower = {s.lower(): s for s in jd_skills}
        
        matched_lower = set(res_skills_lower.keys()).intersection(set(jd_skills_lower.keys()))
        missing_lower = set(jd_skills_lower.keys()) - set(res_skills_lower.keys())
        extra_lower = set(res_skills_lower.keys()) - set(jd_skills_lower.keys())
        
        matched_skills = [jd_skills_lower[k] for k in matched_lower]
        missing_skills = [jd_skills_lower[k] for k in missing_lower]
        extra_skills = [res_skills_lower[k] for k in extra_lower]
        
        return {
            'Matched_Skills': matched_skills,
            'Missing_Skills': missing_skills,
            'Extra_Skills': extra_skills,
            'Match_Percentage': self.calculate_ats_score(resume_skills, jd_skills)
        }

if __name__ == "__main__":
    metrics = IndustryMetrics()
    res_skills = ['Python', 'Java', 'SQL', 'Communication']
    jd_skills = ['Python', 'Machine Learning', 'SQL', 'AWS']
    print(metrics.skill_gap_analysis(res_skills, jd_skills))
