# Medical Insurance Cost Prediction

A machine learning project that predicts individual medical insurance charges from personal and demographic attributes, using exploratory data analysis, feature engineering, and statistical feature selection in Python.

## Overview

Insurance companies price policies based on risk. This project explores which factors (age, BMI, smoking status, number of children, region, sex) drive medical charges, and builds a regression model to estimate them.

## Dataset

- **File:** `insurance.csv`
- **Size:** 1,338 records, 7 columns (1,337 after removing one duplicate row)
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- **Target:** `charges` (individual medical costs billed by health insurance)
- No missing values were found.

## Workflow

### 1. Exploratory Data Analysis
- Inspected shape, data types, summary statistics, and missing values.
- Plotted histograms (with KDE) for `age`, `bmi`, `children`, and `charges`.
- Used box plots to check for outliers.
- Used count plots for `children`, `sex`, and `smoker`.
- Built a correlation heatmap for numeric features.

### 2. Data Cleaning and Preprocessing
- Removed duplicate rows.
- Encoded `sex` as `is_female` (0/1) and `smoker` as `is_smoker` (0/1).
- One-hot encoded `region` (`drop_first=True` to avoid multicollinearity).
- Standardized `age`, `bmi`, and `children` with `StandardScaler`.

### 3. Feature Engineering
- Created a `bmi_category` feature (Underweight, Normal, Overweight, Obese) from standard BMI ranges, then one-hot encoded it.

### 4. Feature Selection
- **Pearson correlation** between each feature and `charges`. `is_smoker` was the strongest predictor (about 0.79), followed by `age` (about 0.30) and BMI-related features.
- **Chi-square test of independence** (alpha = 0.05) between categorical features and quartile-binned `charges`. Features kept: `is_smoker`, `region_southeast`, `is_female`, `bmi_category_Obese`. Features dropped: `region_northwest`, `region_southwest`, `bmi_category_Normal`, `bmi_category_Overweight`.

**Final feature set:** `age`, `is_female`, `bmi`, `children`, `is_smoker`, `region_southeast`, `bmi_category_Obese`

### 5. Modeling
- Trained a linear regression model with scikit-learn on the final feature set.
- Result: R² of about 0.80 on the test set. <!-- TODO: confirm this number against your notebook before publishing -->

## Key Findings
- Smoking status is by far the strongest driver of insurance charges.
- Age has a moderate positive relationship with charges.
- Obesity (BMI of 30 or more) is associated with higher charges.
- Region and sex have only a weak relationship with charges.

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, SciPy, Jupyter Notebook

## How to Run

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
jupyter notebook
```

Open the notebook and run all cells. Make sure `insurance.csv` is in the same folder as the notebook.

## Possible Improvements
- Compare against Ridge, Lasso, Random Forest, and Gradient Boosting models.
- Add an interaction feature (such as smoker x BMI), since the smoker effect looks stronger at high BMI.
- Apply a log transform to `charges` to reduce skew.
- Use cross-validation and report MAE and RMSE alongside R².
- Fit the scaler on the training set only, to avoid data leakage.


