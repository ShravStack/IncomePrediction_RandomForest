# Income Prediction using Random Forest Classifier

A machine learning project that predicts whether an individual's annual income exceeds $50K using U.S. Census demographic and employment data.

## Overview
Built a binary classification pipeline using a Random Forest Classifier trained on the UCI Adult Census dataset. The project covers a complete end-to-end data science workflow: exploratory data analysis, preprocessing, model training, evaluation, and feature interpretation.

## Tech Stack
Python · scikit-learn · pandas · NumPy · seaborn · matplotlib

## Exploratory Data Analysis
The notebook includes a full EDA section covering:
- Dataset shape, data types, and summary statistics
- Missing value analysis (originally encoded as ? in raw data)
- Target class distribution: dataset is imbalanced (~75% earn ≤$50K)
- Age distribution by income class
- Education level vs income countplot
- Occupation vs income breakdown
- Hours per week vs income boxplot
- Correlation heatmap across all numerical features

## Model
**Random Forest Classifier** 
## Key Results
- Evaluated using **accuracy**, **classification report** (precision, recall, F1), and **confusion matrix**
- Top predictors identified: capital.gain, education.num, age, hours.per.week, and occupation/marital status categories
- Feature importances extracted correctly after One-Hot Encoding expansion

## Project Structure
```
├── Income_Prediction_RandomForest.ipynb
└── gl_census_data.csv
```
