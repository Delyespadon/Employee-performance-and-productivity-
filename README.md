# Employee-performance-and-productivity-analysis-using-python

## About the data set
This dataset contains 100,000 rows of data capturing key aspects of employee performance, productivity, and demographics in a corporate environment. It includes details related to the employee's job, work habits, education, performance, and satisfaction. The dataset is designed for various purposes such as HR analytics, employee churn prediction, productivity analysis, and performance evaluation.

Employee_ID: Unique identifier for each employee.
Department: The department in which the employee works (e.g., Sales, HR, IT).
Gender: Gender of the employee (Male, Female, Other).
Age: Employee's age (between 22 and 60).
Job_Title: The role held by the employee (e.g., Manager, Analyst, Developer).
Hire_Date: The date the employee was hired.
Years_At_Company: The number of years the employee has been working for the company.
Education_Level: Highest educational qualification (High School, Bachelor, Master, PhD).
Performance_Score: Employee's performance rating (1 to 5 scale).
Monthly_Salary: The employee's monthly salary in USD, correlated with job title and performance score.
Work_Hours_Per_Week: Number of hours worked per week.
Projects_Handled: Total number of projects handled by the employee.
Overtime_Hours: Total overtime hours worked in the last year.
Sick_Days: Number of sick days taken by the employee.
Remote_Work_Frequency: Percentage of time worked remotely (0%, 25%, 50%, 75%, 100%).
Team_Size: Number of people in the employee's team.
Training_Hours: Number of hours spent in training.
Promotions: Number of promotions received during their tenure.
Employee_Satisfaction_Score: Employee satisfaction rating (1.0 to 5.0 scale).
Resigned: Boolean value indicating if the employee has resigned

## Objective
-	Analyze qualitative-qualitative, qualitative-quantitative, and quantitative-quantitative relationships that will help in optimizing the Employees  performance and the satisfaction.
## Dataset

The data for this project is sourced from the Kaggle dataset:

- **Dataset Link:**(https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data)
  ## 1) Libraries importation
```python
# Import necessary libraries
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.stats as stats
import seaborn as sns
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats import shapiro
 ```
## 2) Data set loading  and exploration
```python
# Loading of the dataset 
df = pd.read_csv(r"C:\Users\Espadon\Desktop\PortfolioProjects-main\Extended_Employee_Performance_and_Productivity_Data.csv", encoding='ISO-8859-1')
df.head(10)
```
```python
# data exploration 
print(df.shape)
print(df.columns)
print(df.dtypes)
df.head()
```
## 3) Data cleaning and Preprocessing
```python
# Checking for duplicated values
print("Number of duplicated rows:", df.duplicated().sum())
```
```python
# Dropping duplicates
df = df.drop_duplicates()
```
```python
# Checking for missing values
print("\nMissing values per column:", df.isna().sum())
```
```python
# Convert date column
df["Hire_Date"] = pd.to_datetime(df["Hire_Date"], errors='coerce')
```
```python
# set "Employee_ID" as index
df = df.set_index("Employee_ID")
```
```python
# Standardize column names
df.columns = df.columns.str.lower()
```
```python
# transform performance level into news variables 
def performance_level(score):
    if score <= 2:
        return 'Low'
    elif score == 3:
        return 'Medium'
    else:
        return 'High'

df['Performance_Level'] = df['performance_score'].apply(performance_level)
```
```python
# Categorize "remote_work_frequency"
df['remote_category'] = df['remote_work_frequency'].map({
    0: 'Never',
    25: 'Rarely',
    50: 'Sometimes',
    75: 'Often',
    100: 'Always'
})
```
## 4) Exploratory Data Analysis(EDA)
### Demographic
```python
# Total number of employees 
print("Total Employees:", df.shape[0])
```
```python
# Employees distribution  by gender
employees_by_gender = df["gender"].value_counts(normalize= True)
print("Employees distribution  by gender\n", employees_by_gender)
```
```python
# Employees distribution  by educational level
employees_by_education = df['education_level'].value_counts(normalize= True)
print(" Employees distribution  by Marital status\n", employees_by_education )
```
### Basic analysis 
```python
# Employees distribution  by department 
employees_by_department = df["department"].value_counts(normalize= True)
print("Employees distribution  by department \n", employees_by_department)
```
```python
# Employees distribution  by job title
employees_by_job = df['job_title'].value_counts(normalize= True)
print(" Employees distribution  by job title\n", employees_by_job )
```
```python
# employee distribution by working style 
employees_by_work = df['remote_category'].value_counts(normalize= True)
print(" employee distribution by working stylee\n", employees_by_work )
```
### Descriptive analysis 
```python
df.describe()
```
## 5) Performance analysis 
### Exploratory analysis 
```python
# Performance by gender
performance_by_gender = pd.crosstab(df['gender'], df['Performance_Level']) 
print("performance by department\n", performance_by_gender)

#Performance by department 
performance_by_department = pd.crosstab(df['department'], df['Performance_Level'])
print("Average performance by department\n", performance_by_department)

# Performance by job title
performance_by_job_title = pd.crosstab(df['job_title'], df['Performance_Level'])
print("Average performance by job title\n", performance_by_job_title)

# Performance by education level
performance_by_education = pd.crosstab(df['education_level'], df['Performance_Level'])
print("Average performance by education level\n", performance_by_education)

# performance by remote category
performance_by_remote_category = pd.crosstab(df['remote_category'], df['Performance_Level'])
print("Performance by remote category\n", performance_by_remote_category)

# Performance by resignation 
performance_by_resignation = pd.crosstab(df['resigned'], df['Performance_Level'])
print("Performance by remote category\n", performance_by_resignation)
```
### Chisquare analysis 
```python
# Chi square analysis of the performance related to other varaiables
from scipy.stats import chi2_contingency
# test if the performance_level  is function of other categorical variable 
cat_col = df.select_dtypes(include = ["object", "bool"]).columns
final_cat_col = cat_col.drop(["Performance_Level"])
target = "Performance_Level"
df_clean = df[[*final_cat_col, target]]

chi2_results = []
for col in final_cat_col:
    contingency_table = pd.crosstab(df_clean[col], df_clean[target])
    if contingency_table.shape[0] >1 and contingency_table.shape[1]>1:
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        chi2_results.append({
            "variable": col,
            "Chi2 Statistic": chi2,
            "p-value": p,
            "Degree of Freedom": dof})
result = pd.DataFrame(chi2_results).sort_values(by ="p-value")

print(result)
```
### Performance analysis with other quantitative variables 
```python
# Select numerical variables
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
# Exclude target if accidentally in num_cols
num_cols = num_cols.drop(['performance_score',"remote_work_frequency"])
```
```python
# test for normality  
for col in num_cols:
    stat, p_value = shapiro(df[col])
    print(f"{col}: p-value = {p_value:.4f} (Normal: {'Yes' if p_value > 0.05 else 'No'})")
```
```python
# Boxplots for visual comparison ===
print("=== Boxplots ===")
for col in num_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Performance_Level', y=col, data=df)
    plt.title(f"{col} by Performance Level")
    plt.tight_layout()
    plt.show()
```
```python
# Run Kruskal-Wallis for each numerical variable
results = {}
for col in num_cols:
    groups = [df[df['Performance_Level'] == dept][col] for dept in df["Performance_Level"].unique()]
    stat, p_value = kruskal(*groups)
    results[col] = {'H-statistic': stat, 'p-value': p_value}

# Convert results to a DataFrame for better readability
import pandas as pd
results_df = pd.DataFrame(results).T
results_df['Significant? (p < 0.05)'] = results_df['p-value'] < 0.05
print(results_df.sort_values(by='p-value'))
```
## 5) Satisfaction  analysis 
### Exploratory analysis 
```python
# Boxplots for visual comparison 
print("=== Boxplots ===")
for col in cat_col:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=col, y='employee_satisfaction_score', data=df)
    plt.title(f"{col} by satisfaction Level")
    plt.tight_layout()
    plt.show()
```
```python
# Satisfaction by gender
satisfaction_by_gender = df.groupby("gender")["employee_satisfaction_score"].mean()
print("Satisfaction by gendert\n", satisfaction_by_gender)
```
```python
# Run Kruskal-Wallis for each numerical variable
results = {}
for col in num_cols:
    groups = [df[df['employee_satisfaction_score'] == dept][col] for dept in df["employee_satisfaction_score"].unique()]
    stat, p_value = kruskal(*groups)
    results[col] = {'H-statistic': stat, 'p-value': p_value}

# Convert results to a DataFrame for better readability
import pandas as pd
results_df = pd.DataFrame(results).T
results_df['Significant? (p < 0.05)'] = results_df['p-value'] < 0.05
print(results_df.sort_values(by='p-value'))
```
## III) Keys Insights of the analysis 
With the first part of the analysis( Python) we observed that: 
- There are 100,000 employees and both gender( male, female ) were equaly represented 
- Employees holding bachelor degree  are  the most represented(50%) while those with PhD  the least represented with 5.4%.
- the employees age were between 20 to  70 year with an average of 41 years a  stnadard deviation of 11 years.
- The average performnce score and monthly salary are 3 and $6,403 respectively.
- After using chisquare analysis the performance are influenced by the gender and the educational level with respective 0.01 and 0.013 pvalues
- Moreover the  performance is also influenced by other factors like the monthly salary, the team size and the number of sicks days.
-  Concerning the satisfaction score , it was equaly distributed accross all every varaibles and was not influenced by factors

  ###  IV) Recommendations
 1) Targeted engagement in low-scoring units
 Customer Support needs priority: run pulse surveys, listening sessions, and fast-track fixes (workload balance, clearer career paths, upgraded tools).
 Marketing and Legal are only marginally better—consider cross-training and recognition programs to raise both morale and output.
 2)Replicate success practices from top units
 Operations and IT policies (e.g., strong mentorship, agile workflows, regular up-skilling) should be documented and shared; pilot these in other teams.
 3)Align incentives where satisfaction ≠ performance
 Finance employees feel good but deliver average results. Introduce performance-based bonuses linked to project outcomes and provide coaching on goal-setting.
 4) Continuous feedback loops: Deploy quarterly satisfaction/performance dashboards at team level so managers can act before issues snowball.
 5) Skill-building and career growth: Company-wide learning budget: tie training hours to clear competency maps; require post-training application projects so gains show up in performance scores.
6) Flexible work & wellness programs: Data show no major downside from remote work frequency, so expand hybrid options and promote wellness days to pre-empt burnout, especially in high-overtime roles.

   
 ### v) Project Publishing and Documentation
   - **Documentation**: Maintain well-structured documentation of the entire process in Markdown 
   - **Project Publishing**: Publish the completed project on GitHub or any other version control platform, including:
     - The `README.md` file (this document).
     - Jupyter Notebooks (if applicable).
     - The Dashboard.
     - Data files (if possible) or steps to access them.
  ## Requirements
- **Python Programming Language**: python editors

- **Kaggle API Key** (for data downloading)

## Getting Started

```
 Set up your Kaggle API, download the data, and follow the steps to load and analyze.
---

## Project Structure

```plaintext
|-- data/                     # Raw data and transformed data
|-- Python_script/              # Python scripts for analysis and queries
|-- README.md                 # Project documentation
|-- main.py                   # Main script for loading, cleaning, and processing data
```

## Contributing

Contributions are welcome!

## License

This project is licensed under the MIT License.

## Acknowledgments

- **Data Source**: Kaggle’s Employee Performance and Productivity Data
