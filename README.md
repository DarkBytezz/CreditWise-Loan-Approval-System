# 💳 CreditWise Loan Approval System

Started this project to explore how different ML models behave on structured financial data instead of just training models blindly for accuracy.

The idea was simple:

> Can we predict loan approvals while minimizing risky approvals?

Because in real loan systems:
- approving a risky customer is dangerous
- rejecting a worthy customer is unfortunate
- but False Positives hurt the most

So precision became one of the main focus metrics.

---

# 🧹 What I Explored

- Missing value handling
- Median vs mode imputation
- Feature engineering
- Correlation analysis
- Label encoding vs one-hot encoding
- Scaling
- Class imbalance
- Precision vs Recall tradeoffs
- Confusion matrix interpretation
- Hyperparameter tuning

Also discovered pretty quickly that:
> tabular financial datasets behave VERY differently from toy ML datasets.

---

# 🧠 Day 1 Models

## Logistic Regression
Most balanced model overall.

| Metric | Score |
|---|---|
| Accuracy | 82.56% |
| Precision | 78.46% |
| Recall | 62.20% |

---

## Gaussian Naive Bayes
Unexpectedly gave the best precision initially.

| Metric | Score |
|---|---|
| Accuracy | 83.33% |
| Precision | 88.24% |
| Recall | 54.88% |

This was interesting because:
- fewer risky customers were approved
- but many worthy customers were still rejected

Classic finance tradeoff.

---

## KNN
Performed noticeably weaker on this dataset.

Main takeaway:
> KNN struggles badly with structured high-dimensional financial data.

---

# 🌲 Day 2: Decision Trees

This completely changed the project.

After experimenting with:
- pruning
- entropy vs gini
- max depth
- leaf size
- GridSearchCV

the Decision Tree massively outperformed every previous model.

Optimized specifically for precision.

### Final Result

| Metric | Score |
|---|---|
| Accuracy | 95.35% |
| Precision | 91.67% |
| Recall | 93.90% |
| F1 Score | 92.77% |

Best Params:
```python
{
    'criterion': 'entropy',
    'max_depth': 7,
    'min_samples_leaf': 4,
    'ccp_alpha': 0.0056
}
```

Only:
- 7 false approvals
- 5 false rejections

which honestly surprised me.

---

# 📌 Biggest Realization

Tree-based models dominate structured/tabular datasets because they naturally capture:
- nonlinear relationships
- threshold logic
- feature interactions

Things like:
```python
if credit_score > X
and DTI < Y
and income > Z
```

are exactly what Decision Trees thrive at.

---

# 🚀 Next

- Random Forest
- XGBoost
- Streamlit deployment
- SHAP explainability

Still exploring.

---

Built while trying to understand *why* models behave the way they do instead of treating sklearn like a magical accuracy vending machine.