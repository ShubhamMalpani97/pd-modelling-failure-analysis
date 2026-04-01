# PD Modelling on German Credit Data -- A Documented Failure Exercise

## Overview

This project builds a complete Probability of Default modelling pipeline on the German Credit dataset. The model fails to produce meaningful discrimination. This is intentional and the failure is fully documented.

The purpose of this project is not to demonstrate a working model. It is to demonstrate the ability to build a technically correct pipeline, diagnose why it fails, and understand the difference between a model that fails because of bad implementation and one that fails because of bad data.

---

## Repository Structure

```
pd-modelling-failure-analysis/
    German_Credit_data.ipynb       # Full working notebook
    german_credit_data.csv         # Source dataset
    eda_numeric_distributions.png  # EDA plot
    requirements.txt               # Python dependencies
    README.md
```

---

## Dataset

The German Credit dataset contains 1,000 borrowers with 9 features: Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration, and Purpose.

The dataset does not contain genuine default outcomes. A binary target variable (Risk) was simulated using Bernoulli trials with feature-adjusted default probabilities. Base probability was set at 0.20 with increments applied for low savings, low checking account balance, long loan duration, high credit amount, and unskilled employment. Probabilities were clipped at 0.85.

This simulation is the root cause of the model's failure and is discussed in detail below.

---

## Pipeline Summary

### Step 1: Data Simulation

Bernoulli trials with adjusted probabilities. Final default rate: 41.3%. The high default rate relative to real retail portfolios (typically 1-5%) is a consequence of the simulation design.

### Step 2: Exploratory Data Analysis

Missing values identified in Saving accounts (183) and Checking account (394). Default rates computed by category for all categorical features. Numeric distributions plotted by default status.

### Step 3: Preprocessing

- Missing values in categorical columns replaced with `unknown` as a standalone bin rather than imputed. This preserves the information content of missingness.
- Small categories in Purpose merged based on two criteria: cell size below 50 and similar default rates.
- Credit amount and Duration log-transformed using log1p to address right skew and satisfy the linearity assumption in logistic regression.
- Job converted from integer to string to prevent the model from treating it as a continuous ordinal quantity.

### Step 4: WOE and IV

Weight of Evidence computed for all categorical and binned continuous features.

- WOE formula: `log(% non-defaults / % defaults)`
- IV formula: `sum of (% non-defaults - % defaults) x WOE` across all bins

**IV Results:**

| Feature | IV | Interpretation |
|---|---|---|
| log_credit_amount | 0.1136 | Medium |
| Saving accounts | 0.0586 | Weak |
| log_duration | 0.0543 | Weak |
| Checking account | 0.0539 | Weak |
| Purpose | 0.0394 | Weak |
| Age | 0.0237 | Weak |
| Job | 0.0034 | Useless |
| Housing | 0.0007 | Useless |

Job and Housing were dropped before modelling. No feature cleared the strong threshold of 0.30. This was the primary signal that the model would underperform.

### Step 5: Train-Test Split

80/20 split stratified on the target variable to preserve the default rate in both sets.

- Train default rate: 41.2%
- Test default rate: 41.5%

WOE encoder fitted on the training set only and applied to both sets. Fitting on the full dataset would constitute data leakage.

### Step 6: Logistic Regression

L2 regularisation. Regularisation parameter C selected via 5-fold cross-validation optimising AUROC across a grid of `[0.001, 0.01, 0.1, 1.0, 10.0]`. Cross-validation selected C=1.0, the weakest regularisation in the grid. This indicated that stronger regularisation was removing genuine signal rather than suppressing noise -- consistent with the low IV features.

**Model Coefficients:**

| Feature | Coefficient |
|---|---|
| log_duration_bin_WOE | -0.2132 |
| Purpose_WOE | -0.8224 |
| Saving accounts_WOE | -0.8249 |
| Checking account_WOE | -0.9060 |
| log_credit_amount_bin_WOE | -0.9073 |
| Age_bin_WOE | -0.9107 |

All coefficients are negative. This is expected and correct in a WOE-based model. WOE is defined as `log(non-defaults / defaults)`, so higher WOE means safer. A negative coefficient preserves the correct direction: as WOE increases (bin becomes safer), the log-odds of default decrease.

### Step 7: Model Evaluation

| Metric | Train | Test |
|---|---|---|
| AUROC | 0.657 | 0.548 |
| Gini | 0.313 | 0.096 |
| KS | 0.242 | 0.119 |

Train-test Gini gap: 21 points. Standard threshold for concern is 3-4 points. This gap indicates overfitting -- the model learned patterns specific to the 800 training borrowers that did not generalise to the 200 test borrowers.

Test Gini of 0.096 is near-random. Industry minimum for an acceptable scorecard is typically 0.25-0.30.

### Step 8: Scorecard Scaling

Business parameters: PDO=20, Base Score=600, Base Odds=1.

```
Factor = PDO / ln(2) = 28.85
Offset = Base Score - Factor x ln(Base Odds) = 600
```

Points per bin computed as: `-(coefficient x WOE x Factor)`

**Score Distribution Across 200 Test Borrowers:**

| Statistic | Value |
|---|---|
| Mean | 628.9 |
| Std | 16.6 |
| Min | 592 |
| Max | 664 |

**Score Separation Between Risk Groups:**

| Group | Mean Score |
|---|---|
| Non-defaulters | 630.0 |
| Defaulters | 627.4 |

Separation of 2.6 points against a standard deviation of 16-18 points. The scorecard cannot operationally distinguish defaulters from non-defaulters.

---

## Root Cause Analysis

The model failure has one primary cause: the target variable was simulated.

Genuine default outcomes carry information that propagates through all features. Real defaulters differ from non-defaulters in ways that are observable in the data and captured by features. A simulated target based on additive probability rules produces features with low IV by construction because the simulation rules were deliberately simple.

This is a data problem, not a modelling problem. Every technical step in the pipeline was implemented correctly. The WOE encoding, the train-test discipline, the cross-validation, the scorecard scaling -- all correct.

---

## Key Lessons

**Lesson 1: IV before modelling**

Information Value is a pre-modelling diagnostic. If no feature clears IV of 0.10, the model will underperform regardless of the algorithm chosen. IV told this story completely before the logistic regression was even fitted.

**Lesson 2: Cross-validation selecting weak regularisation is a signal**

When cross-validation selects the weakest regularisation in the grid, the interpretation is not that the model needs no regularisation. It means the features are so weak that stronger regularisation removes genuine signal. This points back to data quality, not model tuning.

**Lesson 3: Train-test gap diagnosis**

A 21-point Gini gap is overfitting. The correct response is not to tune harder but to understand why. In this case: weak features that allow the model to find spurious patterns in 800 training rows that disappear on 200 test rows.

**Lesson 4: Synthetic data has a ceiling**

No pipeline improvement can overcome a fundamentally weak target variable. The correct response when facing weak real data is feature engineering, additional data sources, or a different target definition -- not model complexity.

**Lesson 5: A correct pipeline on bad data still produces bad outputs**

This is the most practically important lesson. In production risk modelling, technically correct implementation is necessary but not sufficient. Data quality, target variable reliability, and feature signal are the binding constraints.

---

## What Would Make This Model Work

A real dataset with genuine historical default outcomes. Features with IV above 0.10. A default rate representative of the actual portfolio (typically 1-5% in retail credit). With those inputs, this exact pipeline would produce a scorecard with Gini above 0.30 and meaningful score separation between risk groups.

---

## Dependencies

```
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
```

Install all dependencies with:

```
pip install -r requirements.txt
```

---

## Author

Shubham -- Quantitative Credit Risk Analyst
