---
layout: home
author_profile: true
header:
  overlay_color: "#5e616c"
  overlay_image: /assets/images/header-bg.jpg
  actions:
    - label: "View Resume"
      url: "/assets/documents/C_Rudder_Gen_CV_USA.pdf"
    - label: "Contact Me"
      url: "mailto:carla.amoi@gmail.com"
excerpt: >
  Data-driven professional with expertise in Python, SQL, and machine learning.<br />
  Transforming complex data into actionable business insights.
feature_row:
  - image_path: /assets/images/foodhub-thumb.png
    alt: "FoodHub Analysis"
    title: "FoodHub Business Analysis"
    excerpt: "Analyzed 1,800+ food delivery orders to optimize operations and increase revenue. Identified key drivers of customer satisfaction and provided actionable recommendations."
    url: "/portfolio/foodhub-analysis/"
    btn_class: "btn--primary"
    btn_label: "View Project"
    tags: "Python • Data Analysis • Business Intelligence"
  - image_path: /assets/images/extralearn-thumb.png
    alt: "ExtraaLearn Prediction"
    title: "Customer Conversion Prediction"
    excerpt: "Built machine learning models (Decision Tree, Random Forest) to predict lead conversion with 85% accuracy. Identified key factors driving customer acquisition."
    url: "/portfolio/extralearn-prediction/"
    btn_class: "btn--primary"
    btn_label: "View Project"
    tags: "Machine Learning • Classification • Python"
  - image_path: /assets/images/loan-thumb.png
    alt: "Loan Default"
    title: "Loan Default Prediction"
    excerpt: "Developed classification models to predict loan defaults, helping banks make data-driven lending decisions. Feature importance analysis for risk assessment."
    url: "/portfolio/loan-default-prediction/"
    btn_class: "btn--primary"
    btn_label: "View Project"
    tags: "Machine Learning • Risk Analysis • Python"
---

<style>
/* Custom color palette badges */
.skills-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 2rem;
  margin: 2rem 0;
}

.skills-category {
  background-color: #e0e1dd;
  padding: 1.5rem;
  border-radius: 8px;
  border-left: 4px solid #415a77;
}

.skills-category h3 {
  color: #1b263b;
  margin-top: 0;
  margin-bottom: 1rem;
  font-size: 1.1rem;
}

.badge-container {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

/* Core competencies with icons */
.competencies-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.competency-card {
  background-color: #e0e1dd;
  padding: 1.5rem;
  border-radius: 8px;
  border-top: 3px solid #415a77;
  transition: transform 0.2s;
}

.competency-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 12px rgba(65, 90, 119, 0.2);
}

.competency-icon {
  font-size: 2rem;
  margin-bottom: 0.5rem;
  color: #415a77;
}

.competency-title {
  color: #1b263b;
  font-weight: bold;
  margin-bottom: 0.5rem;
  font-size: 1.1rem;
}

.competency-skills {
  color: #415a77;
  font-size: 0.9rem;
  line-height: 1.6;
}

/* Education cards with logos - SMALLER */
.education-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
  margin: 2rem 0;
}

.education-card {
  background-color: #e0e1dd;
  padding: 1rem 1.25rem;
  border-radius: 8px;
  border-left: 4px solid #415a77;
  display: flex;
  align-items: center;
  gap: 1.25rem;
  transition: transform 0.2s;
}

.education-card:hover {
  transform: translateX(5px);
  box-shadow: 0 4px 12px rgba(65, 90, 119, 0.2);
}

.education-logo {
  width: 60px;
  height: 60px;
  object-fit: contain;
  flex-shrink: 0;
}

.education-content {
  flex-grow: 1;
}

.education-degree {
  color: #1b263b;
  font-weight: bold;
  font-size: 1rem;
  margin-bottom: 0.2rem;
}

.education-school {
  color: #415a77;
  font-size: 0.9rem;
  margin-bottom: 0.2rem;
}

.education-details {
  color: #778da9;
  font-size: 0.8rem;
  font-style: italic;
}

/* Certifications in 2x3 grid */
.certifications-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin: 2rem 0;
}

.cert-item {
  background-color: #e0e1dd;
  padding: 1rem 1.25rem;
  border-radius: 8px;
  border-left: 3px solid #415a77;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: all 0.2s;
}

.cert-item:hover {
  background-color: #778da9;
  transform: translateX(5px);
}

.cert-name {
  color: #1b263b;
  font-weight: 600;
  font-size: 0.9rem;
}

.cert-issuer {
  color: #415a77;
  font-size: 0.8rem;
  margin-top: 0.2rem;
}

.cert-link {
  background-color: #415a77;
  color: #e0e1dd;
  padding: 0.4rem 0.8rem;
  border-radius: 5px;
  text-decoration: none;
  font-size: 0.8rem;
  transition: background-color 0.2s;
  white-space: nowrap;
}

.cert-link:hover {
  background-color: #1b263b;
  color: #e0e1dd;
  text-decoration: none;
}

@media (max-width: 768px) {
  .skills-grid {
    grid-template-columns: 1fr;
  }
  
  .certifications-grid {
    grid-template-columns: 1fr;
  }
  
  .education-card {
    flex-direction: column;
    text-align: center;
  }
  
  .education-logo {
    width: 50px;
    height: 50px;
  }
  
  .cert-item {
    flex-direction: column;
    gap: 0.75rem;
    text-align: center;
  }
}
</style>

## About Me

I'm a **data analyst** and **PMP-certified project manager** with 15+ years of experience leading data-driven initiatives in education and international research. I combine advanced quantitative skills (Python, SQL, machine learning) with full-cycle project management expertise to deliver measurable business value.

Recently completed MIT's **Applied Data Science and Machine Learning** certificate, I'm passionate about applying analytics to healthcare and business intelligence challenges.

### Core Competencies

<div class="competencies-grid">
  <div class="competency-card">
    <div class="competency-icon">📊</div>
    <div class="competency-title">Data Science & Analytics</div>
    <div class="competency-skills">Python (Pandas, NumPy, scikit-learn), SQL, Statistical Analysis</div>
  </div>
  
  <div class="competency-card">
    <div class="competency-icon">🤖</div>
    <div class="competency-title">Machine Learning</div>
    <div class="competency-skills">Classification, Regression, Predictive Modeling, Feature Engineering</div>
  </div>
  
  <div class="competency-card">
    <div class="competency-icon">📈</div>
    <div class="competency-title">Visualization</div>
    <div class="competency-skills">Matplotlib, Seaborn, Tableau, Power BI</div>
  </div>
  
  <div class="competency-card">
    <div class="competency-icon">🎯</div>
    <div class="competency-title">Project Management</div>
    <div class="competency-skills">PMP Certified, Agile/Scrum (CSM), Full Project Lifecycle</div>
  </div>
  
  <div class="competency-card">
    <div class="competency-icon">💼</div>
    <div class="competency-title">Business Intelligence</div>
    <div class="competency-skills">Requirements Gathering, Stakeholder Engagement, Executive Communication</div>
  </div>
</div>

---

## Featured Projects

{% include feature_row %}

---

## Technical Skills

<div class="skills-grid">
  <div class="skills-category">
    <h3>Programming & Analysis</h3>
    <div class="badge-container">
      <img src="https://img.shields.io/badge/Python-415a77?style=for-the-badge&logo=python&logoColor=e0e1dd" alt="Python">
      <img src="https://img.shields.io/badge/SQL-1b263b?style=for-the-badge&logo=postgresql&logoColor=e0e1dd" alt="SQL">
      <img src="https://img.shields.io/badge/R-415a77?style=for-the-badge&logo=r&logoColor=e0e1dd" alt="R">
    </div>
  </div>

  <div class="skills-category">
    <h3>Machine Learning & Data Science</h3>
    <div class="badge-container">
      <img src="https://img.shields.io/badge/scikit--learn-415a77?style=for-the-badge&logo=scikit-learn&logoColor=e0e1dd" alt="scikit-learn">
      <img src="https://img.shields.io/badge/Pandas-1b263b?style=for-the-badge&logo=pandas&logoColor=e0e1dd" alt="Pandas">
      <img src="https://img.shields.io/badge/NumPy-415a77?style=for-the-badge&logo=numpy&logoColor=e0e1dd" alt="NumPy">
    </div>
  </div>

  <div class="skills-category">
    <h3>Visualization & BI</h3>
    <div class="badge-container">
      <img src="https://img.shields.io/badge/Tableau-415a77?style=for-the-badge&logo=tableau&logoColor=e0e1dd" alt="Tableau">
      <img src="https://img.shields.io/badge/Power_BI-1b263b?style=for-the-badge&logo=powerbi&logoColor=e0e1dd" alt="Power BI">
      <img src="https://img.shields.io/badge/Matplotlib-415a77?style=for-the-badge&logo=python&logoColor=e0e1dd" alt="Matplotlib">
    </div>
  </div>

  <div class="skills-category">
    <h3>Tools & Platforms</h3>
    <div class="badge-container">
      <img src="https://img.shields.io/badge/Jupyter-415a77?style=for-the-badge&logo=jupyter&logoColor=e0e1dd" alt="Jupyter">
      <img src="https://img.shields.io/badge/Git-1b263b?style=for-the-badge&logo=git&logoColor=e0e1dd" alt="Git">
      <img src="https://img.shields.io/badge/Excel-415a77?style=for-the-badge&logo=microsoft-excel&logoColor=e0e1dd" alt="Excel">
    </div>
  </div>
</div>

---

## Education & Certifications

### Education

<div class="education-grid">
  <div class="education-card">
    <img src="https://logos-world.net/wp-content/uploads/2021/10/MIT-Logo.png" alt="MIT Logo" class="education-logo">
    <div class="education-content">
      <div class="education-degree">Applied Data Science & Machine Learning</div>
      <div class="education-school">Massachusetts Institute of Technology</div>
      <div class="education-details">Professional Certificate Program</div>
    </div>
  </div>

  <div class="education-card">
    <img src="https://1000logos.net/wp-content/uploads/2022/08/Florida-State-University-Logo.png" alt="FSU Logo" class="education-logo">
    <div class="education-content">
      <div class="education-degree">Ph.D. Mathematics Education</div>
      <div class="education-school">Florida State University</div>
      <div class="education-details">Doctoral Degree</div>
    </div>
  </div>

  <div class="education-card">
    <img src="https://1000logos.net/wp-content/uploads/2022/08/Florida-State-University-Logo.png" alt="FSU Logo" class="education-logo">
    <div class="education-content">
      <div class="education-degree">M.S. Mathematics Education</div>
      <div class="education-school">Florida State University</div>
      <div class="education-details">18+ graduate credits in Applied Mathematics & Statistics</div>
    </div>
  </div>

  <div class="education-card">
    <img src="https://1000logos.net/wp-content/uploads/2022/11/NC-State-Wolfpack-Logo.png" alt="NC State Logo" class="education-logo">
    <div class="education-content">
      <div class="education-degree">B.S. Applied Mathematics</div>
      <div class="education-school">North Carolina State University</div>
      <div class="education-details">Bachelor of Science</div>
    </div>
  </div>
</div>

### Professional Certifications

<div class="certifications-grid">
  <div class="cert-item">
    <div>
      <div class="cert-name">Project Management Professional (PMP)</div>
      <div class="cert-issuer">Project Management Institute</div>
    </div>
    <a href="/assets/documents/certificates/PMP_Certificate.pdf" class="cert-link" target="_blank">View 📄</a>
  </div>

  <div class="cert-item">
    <div>
      <div class="cert-name">Certified Scrum Master (CSM)</div>
      <div class="cert-issuer">Scrum Alliance</div>
    </div>
    <a href="/assets/documents/certificates/CSM_Certificate.pdf" class="cert-link" target="_blank">View 📄</a>
  </div>

  <div class="cert-item">
    <div>
      <div class="cert-name">Applied Data Science & ML Certificate</div>
      <div class="cert-issuer">Massachusetts Institute of Technology</div>
    </div>
    <a href="/assets/documents/certificates/MIT_Certificate.pdf" class="cert-link" target="_blank">View 📄</a>
  </div>

  <div class="cert-item">
    <div>
      <div class="cert-name">Google Data Analytics Professional</div>
      <div class="cert-issuer">Google</div>
    </div>
    <a href="/assets/documents/certificates/Google_Data_Analytics_Certificate.pdf" class="cert-link" target="_blank">View 📄</a>
  </div>

  <div class="cert-item">
    <div>
      <div class="cert-name">IBM Data Science Fundamentals</div>
      <div class="cert-issuer">IBM</div>
    </div>
    <a href="/assets/documents/certificates/IBM_Data_Science_Certificate.pdf" class="cert-link" target="_blank">View 📄</a>
  </div>

  <div class="cert-item">
    <div>
      <div class="cert-name">SQL Basics for Data Science</div>
      <div class="cert-issuer">UC Davis</div>
    </div>
    <a href="/assets/documents/certificates/SQL_Certificate.pdf" class="cert-link" target="_blank">View 📄</a>
  </div>
</div>

---

## Let's Connect

I'm actively seeking opportunities in **data analytics**, **data science**, and **project management** roles, particularly in healthcare analytics and business intelligence.

**Open to:** Full-time positions, contract work, consulting projects

📧 **Email:** [carla.amoi@gmail.com](mailto:carla.amoi@gmail.com)  
💼 **LinkedIn:** [linkedin.com/in/carudder](https://linkedin.com/in/carudder/)  
💻 **GitHub:** [github.com/crud27](https://github.com/crud27)  
📱 **Phone:** +1-984-322-0611

[Download My Resume](/assets/documents/C_Rudder_Gen_CV_USA.pdf){: .btn .btn--primary .btn--large}
