# Employee Attrition Analysis Project

## Overview
This project analyzes employee attrition using a dataset from Kaggle to identify key factors influencing turnover and predict at-risk employees. It employs machine learning techniques, including clustering (K-Means) and classification models (Logistic Regression and Random Forest), implemented in PySpark for handling big data. The goal is to provide actionable insights for HR teams to improve retention strategies, reduce turnover costs, and enhance workforce satisfaction.

## Motivation
Employee attrition leads to high costs in recruitment, training, and productivity loss. By uncovering patterns in attrition (e.g., related to age, job satisfaction, overtime, income, and department), organizations can implement targeted interventions. This project demonstrates a data-driven approach to predict and mitigate attrition.

## Data Description
- **Source**: Kaggle (2020) employee attrition dataset.
- **Size**: 1,470 records with 17 features.
- **Key Features**: Age, Attrition (target), Department, Job Satisfaction, Monthly Income, OverTime, Work-Life Balance, Years at Company, and more.
- **Preprocessing**: Handling missing values (mean imputation for Job Involvement), outlier management (scaling instead of removal for Monthly Income, Years at Company, etc.), and encoding categorical variables (One-Hot, Ordinal, Binary Encoding).

## Project Structure
- **DataPhoenix Group Project Final Report.docx**: Comprehensive report with executive summary, analysis, visualizations, model evaluations, and business recommendations (e.g., for Amazon-like companies).
- **Clustering Analysis For Employee Attrition.ipynb**: PySpark notebook for data cleaning, K-Means clustering, and analysis of clusters (e.g., attrition rates by age, job satisfaction, overtime, income, and department).
- **Classification Models For Employee Attrition.ipynb**: PySpark notebook for data preparation, Logistic Regression (86% accuracy), Random Forest (85% accuracy, F1 0.81), feature importance, and visualizations (scatter/bar plots for key variables vs. attrition).
- **README.md**: This file.

## Methods and Models
- **Clustering**: K-Means to segment employees into clusters based on features like Age, Job Satisfaction, etc. Revealed higher attrition in Cluster 1 (older employees, lower satisfaction).
- **Classification**: 
  - Logistic Regression: 86% accuracy, but weaker on true positives.
  - Random Forest: Selected model (85% accuracy, F1 0.81) for better interpretability and handling class imbalance.
- **Tools**: PySpark (for big data pipelines), Pandas/Matplotlib (visualizations), Scikit-learn (metrics).
- **Key Insights**:
  - Higher attrition among younger employees (<30), low-income groups, and departments like R&D/HR.
  - Job satisfaction and age are stronger drivers than overtime or income.
  - Business recommendations: Mentorship for young talent, competitive pay reviews, department-specific interventions.

## Results
- Clustering: Cluster 1 (higher attrition: 0.043) vs. Cluster 0 (lower: 0.022). Sales has lowest attrition; R&D/HR highest.
- Classification: Random Forest identifies top features (e.g., Age, Monthly Income, Department) for predicting attrition.
- Visualizations: Included in notebooks (e.g., scatter plots for Age vs. Attrition, bar plots for Department vs. Attrition Rate).

## How to Run
1. **Requirements**: Python 3.x, PySpark (install via `pip install pyspark`), Pandas, Matplotlib, Scikit-learn.
2. Clone the repo: `git clone <your-repo-url>`.
3. Run notebooks in Jupyter or Databricks: Open `.ipynb` files and execute cells sequentially.
4. Dataset: Download from Kaggle (search "IBM HR Analytics Employee Attrition") and place in the project root as `HR-Employee-Attrition.csv` (assumed in code).

## Dependencies
- pyspark==3.5.0
- pandas==2.0.0
- matplotlib==3.7.0
- scikit-learn==1.2.0

## Contributors
- DataPhoenix Group (as per report).

## License
MIT License. Feel free to use and modify.

For questions, contact [your-email@example.com].
