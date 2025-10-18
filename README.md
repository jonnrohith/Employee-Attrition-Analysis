# Employee Attrition Analysis Project

## Overview
This project analyzes employee attrition using a dataset from Kaggle to identify key factors influencing turnover and predict at-risk employees. It employs machine learning techniques, including clustering (K-Means) and classification models (Logistic Regression and Random Forest), implemented in PySpark for handling big data. The goal is to provide actionable insights for HR teams to improve retention strategies, reduce turnover costs, and enhance workforce satisfaction.

## Motivation
Employee attrition leads to high costs in recruitment, training, and productivity loss. By uncovering patterns in attrition (e.g., related to age, job satisfaction, overtime, income, and department), organizations can implement targeted interventions. This project demonstrates a data-driven approach to predict and mitigate attrition.

## Data Description
- **Source**: Kaggle (2020) employee attrition dataset.
- **Size**: 1,470 records with 17 features.
- **Key Features**: Age, Attrition (target), Department, Job Satisfaction, Monthly Income, OverTime, Work-Life Balance, Years at Company, and more.

### Dataset Sample
The dataset contains various employee attributes including:
- **Demographics**: Age, Gender, Marital Status
- **Job-related**: Department, Job Satisfaction, Job Involvement, Environment Satisfaction
- **Compensation**: Monthly Income, Percent Salary Hike
- **Work-life factors**: OverTime, Work-Life Balance, Distance from Home
- **Tenure**: Years at Company, Years in Current Role, Years Since Last Promotion, Years with Current Manager

### Data Preprocessing
- **Missing Values**: Handled using mean imputation for Job Involvement
- **Outlier Detection**: Comprehensive analysis revealed outliers in key numerical features:
  - Monthly Income: 114 outliers
  - Years at Company: 104 outliers  
  - Years Since Last Promotion: 107 outliers
  - Years in Current Role: 21 outliers
  - Years with Current Manager: 14 outliers
- **Outlier Management**: Applied scaling instead of removal to preserve data integrity
- **Feature Encoding**: One-Hot, Ordinal, and Binary encoding for categorical variables

### Correlation Analysis
A correlation heatmap revealed important relationships between features:
- **Strong Positive Correlations**: Years at Company with Years in Current Role (0.76), Years with Current Manager (0.77), and Years Since Last Promotion (0.62)
- **Moderate Correlations**: Age with Monthly Income (0.50), Monthly Income with Years at Company (0.51)
- **Weak Correlations**: Most satisfaction metrics showed minimal correlation with other features

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

### Clustering Analysis (K-Means)
The K-Means clustering analysis successfully segmented employees into two distinct groups with significantly different attrition patterns:

#### Cluster Characteristics
| Cluster | Average Age | Attrition Count | Total Employees | Attrition Rate |
|---------|-------------|-----------------|-----------------|----------------|
| 1       | 37.39       | 43,845          | 1,022,847       | 4.29%          |
| 0       | 37.47       | 12,324          | 553,611         | 2.23%          |

**Key Findings:**
- **Cluster 1** shows nearly double the attrition rate (4.29%) compared to Cluster 0 (2.23%)
- Both clusters have similar average ages (~37 years)
- Cluster 1 represents a higher-risk employee segment requiring targeted intervention

#### Detailed Cluster Analysis
| Metric | Cluster 1 | Cluster 0 | Insight |
|--------|-----------|-----------|---------|
| Average Job Satisfaction | 2.77 | 2.77 | Similar satisfaction levels |
| Average Work-Life Balance | 2.78 | 2.78 | Comparable work-life balance |
| Average Environment Satisfaction | 2.76 | 2.76 | Similar environment satisfaction |
| Average Monthly Income | $6,745 | $6,787 | Minimal income difference |
| Average Overtime | 0.247 | 0.241 | Slightly higher overtime in Cluster 1 |

### Classification Model Performance

#### Random Forest Model (Selected Model)
- **Accuracy**: 85%
- **F1 Score**: 0.81
- **RMSE**: 0.38
- **R² Score**: -0.10

#### Confusion Matrix Analysis
```
Confusion Matrix:
        Predicted 0   Predicted 1
Actual 0    205           9
Actual 1    27            13
```

**Performance Breakdown:**
- **True Negatives (TN)**: 205 - Correctly identified employees who stayed
- **False Positives (FP)**: 9 - Incorrectly predicted attrition (Type I errors)
- **False Negatives (FN)**: 27 - Missed employees who actually left (Type II errors)
- **True Positives (TP)**: 13 - Correctly identified employees who left

**Key Observations:**
- Model excels at identifying employees who will stay (high TN count)
- Lower recall for attrition class: 32.5% (13/(13+27))
- Precision for attrition class: 59% (13/(13+9))

### Visual Analysis Results

#### Age vs Attrition Pattern
- **High Attrition in Young Employees**: Peak attrition rate of ~67% at age 18, decreasing to ~30% by mid-20s
- **Lowest Attrition**: Employees in their late 30s to early 40s show minimal attrition
- **Retirement Effect**: Slight increase in attrition for employees approaching retirement (age 58+)

#### Department-wise Attrition Rates
| Department | Attrition Rate |
|------------|----------------|
| Sales | 21.0% |
| Human Resources | 19.0% |
| Research & Development | 14.0% |

**Insights:**
- Sales department shows highest attrition (21%)
- R&D has the lowest attrition rate (14%)
- HR department shows moderate attrition (19%)

### Key Business Insights
1. **Age Factor**: Younger employees (<30) are at highest risk of attrition
2. **Department Risk**: Sales and HR departments require immediate attention
3. **Income Impact**: Lower income correlates with higher attrition risk
4. **Job Satisfaction**: While satisfaction levels are similar across clusters, the combination of factors creates high-risk groups
5. **Overtime Effect**: Slightly higher overtime in high-attrition cluster suggests work-life balance concerns

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
