# 💳 CreditWise Loan Approval System

An end-to-end Machine Learning project that predicts whether a loan application should be approved or rejected based on applicant financial and demographic information.

This project focuses not only on model accuracy, but also on understanding real-world banking tradeoffs such as minimizing risky loan approvals while maintaining reasonable approval rates.

---

# 🚀 Project Overview

Traditional loan approval systems often involve:
- Manual verification
- Human bias
- Slow processing
- Inconsistent decision-making

The goal of this project is to build an intelligent loan approval prediction system using Machine Learning models trained on applicant financial and personal data.

The system performs:
- Data preprocessing
- Missing value handling
- Feature engineering
- Categorical encoding
- Model training
- Evaluation using classification metrics and confusion matrices

---

# 📂 Dataset Features

The dataset contains several applicant-related features:

| Feature | Description |
|---|---|
| Applicant_Income | Primary applicant income |
| Coapplicant_Income | Co-applicant income |
| Employment_Status | Employment type/status |
| Age | Applicant age |
| Marital_Status | Married/Single |
| Dependents | Number of dependents |
| Credit_Score | Applicant credit score |
| Existing_Loans | Existing active loans |
| DTI_Ratio | Debt-to-Income ratio |
| Savings | Applicant savings |
| Collateral_Value | Collateral valuation |
| Loan_Amount | Requested loan amount |
| Loan_Term | Loan duration |
| Loan_Purpose | Purpose of loan |
| Property_Area | Urban/Semi-Urban/Rural |
| Education_Level | Graduate/Not Graduate |
| Employer_Category | Employer classification |
| Loan_Approved | Target variable |

---

# 🛠️ Data Preprocessing

The dataset required extensive preprocessing before training the models.

## ✅ Missing Value Handling

### Dropped rows for critical missing values:
- `Applicant_Income`
- `Loan_Amount`
- `Loan_Approved`

### Median Imputation for Numerical Features:
- Age
- Credit_Score
- Existing_Loans
- DTI_Ratio
- Savings
- Collateral_Value
- Loan_Term
- Dependents

### Mode Imputation for Categorical Features:
- Employment_Status
- Marital_Status
- Loan_Purpose
- Property_Area
- Education_Level
- Gender
- Employer_Category

---

# ⚙️ Feature Engineering

Created a new feature:

```python
Total_Income = Applicant_Income + Coapplicant_Income
```

This helps represent overall household repayment capability more effectively than considering applicant income alone.

Additionally:
- `Applicant_ID` was removed as it had no predictive significance.
- `Coapplicant_Income` was dropped after engineering `Total_Income`.

---

# 🔄 Encoding & Transformation

## Binary Encoding
Mapped binary categorical variables to numeric values:

| Feature | Mapping |
|---|---|
| Loan_Approved | Yes → 1, No → 0 |
| Gender | Male → 1, Female → 0 |
| Marital_Status | Married → 1, Single → 0 |
| Education_Level | Graduate → 1, Not Graduate → 0 |

## One-Hot Encoding
Applied using:
```python
pd.get_dummies()
```

for:
- Employment_Status
- Loan_Purpose
- Property_Area
- Employer_Category

---

# 📊 Data Standardization

Used `StandardScaler` after train-test split to standardize numerical features.

This prevents data leakage and improves model performance for distance-based algorithms like KNN.

---

# 🤖 Machine Learning Models Used

## 1️⃣ Logistic Regression
- LogisticRegressionCV
- Cross-validation enabled
- L2 Regularization

## 2️⃣ Gaussian Naive Bayes

## 3️⃣ K-Nearest Neighbors (KNN)
- Hyperparameter tuning using GridSearchCV

---

# 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 82.56% | 78.46% | 62.20% | 69.39% |
| Naive Bayes | 83.33% | 88.24% | 54.88% | 67.67% |
| KNN | 77.52% | 77.27% | 41.46% | 53.97% |

---

# 🧠 Key Observations

## 🔹 Logistic Regression
Provided the most balanced overall performance across metrics.

## 🔹 Naive Bayes
Achieved the highest precision.

This is important in loan systems because:
- High precision reduces false positives
- Risky customers are less likely to be incorrectly approved

However, lower recall means more genuinely eligible applicants may still be rejected.

## 🔹 KNN
Performed comparatively worse due to:
- High-dimensional tabular data
- Sensitivity to noisy feature spaces

---

# 📉 Confusion Matrix Analysis

Confusion matrices were generated for all models to analyze:
- False Positives
- False Negatives
- Model bias
- Approval/Rejection behavior

This helped evaluate the business impact of predictions rather than relying only on accuracy.

---

# 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# 📁 Project Structure

```bash
CreditWise-Loan-Approval-System/
│
├── loan_approval_prediction.ipynb
├── loan_approval_data.csv
├── README.md
├── requirements.txt
```

---

# ▶️ How to Run

## Clone Repository

```bash
git clone https://github.com/DarkBytezz/CreditWise-Loan-Approval-System.git
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Notebook

Open:

```bash
loan_approval_prediction.ipynb
```

using Jupyter Notebook or VS Code.

---

# 🔮 Future Improvements

- Add Random Forest and XGBoost
- Build Streamlit Web App
- Deploy model online
- Add SHAP Explainability
- Handle class imbalance more effectively
- Add probability-based risk scoring
- Create real-time prediction dashboard

---

# 👨‍💻 Author

### Manan Verma

Passionate about:
- AI/ML
- Backend Systems
- Problem Solving
- Building real-world intelligent systems

---

# ⭐ Final Note

This project focuses not only on Machine Learning implementation, but also on understanding the real-world financial implications of model decisions.

Because in loan systems:

> A model that approves every applicant is accurate for nobody and dangerous for everyone.