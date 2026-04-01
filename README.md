# PD Modelling on German Credit Data -- A Documented Failure Exercise

## Overview

This project builds a complete Probability of Default modelling pipeline on the German Credit dataset. The model fails to produce meaningful discrimination. This is intentional and the failure is fully documented.

The purpose of this project is not to demonstrate a working model. It is to demonstrate two things: the ability to build a technically correct pipeline, and the ability to identify why a model was always going to fail -- not just at the surface level of weak features, but at the deeper level of experimental design.

---

## The Core Flaw -- Read This First

The German Credit dataset contains no genuine default outcomes. A binary target variable (Risk) was simulated using Bernoulli trials with feature-adjusted default probabilities. The simulation used the same features that were later used to model the target.

This makes the entire exercise self-referential.

Any predictive signal the features exhibit is a circular reflection of simulation assumptions, not genuine discriminatory power. The Bernoulli noise introduced during simulation then diluted even that artificial signal, leaving near-zero discrimination in the final model.

The pipeline is technically correct throughout. The experimental design is fundamentally flawed.

This distinction -- between pipeline correctness and experimental validity -- is the central lesson of this project.

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

The dataset does not contain genuine default outcomes. A binary target variable (Risk) was simulated using Bernoulli trials with feature-adjusted default probabilities. Base probability was set at 0.20 with increments applied as follows:

| Condition | Increment |
|---|---|
| Checking account = little | +0.15 |
| Checking account = missing | +0.10 |
| Saving accounts = little | +0.10 |
| Saving accounts = missing | +0.08 |
| Duration > 36 months | +0.10 |
| Duration > 48 months | +0.05 |
| Credit amount > 75th percentile | +0.08 |
| Job = 0 (unskilled, non-resident) | +0.10 |

Probabilities were clipped at 0.85. A Bernoulli draw then converted each probability into a binary outcome.

The features used in this simulation -- savings, checking account, duration, credit amount, job -- are the same features used in the model. This is the experimental design flaw described above.

---

## Pipeline Summary

### Step 1: Data Simulation

Bernoulli trials with adjusted probabilities as described above. Final default rate: 41.3%. The high default rate relative to real retail portfolios (typically 1-5%) is a further consequence of the simulation design.

### Step 2: Exploratory Data Analysis

Missing values identified in Saving accounts (183) and Checking account (394). Default rates computed by category for all categorical features. Numeric distributions plotted by default status.

### Step 3: Preprocessing

- Missing values in categorical columns replaced with `unknown` as a standalone bin. This preserves the information content of missingness rather than imputing it away.
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

No feature cleared the strong threshold of 0.30. Job and Housing were dropped before modelling. The weak IV values appear to show genuine signal but are in fact a diluted circular reflection of the simulation assumptions -- the features were used to build the target and then used to model it.

### Step 5: Train-Test Split

80/20 split stratified on the target variable to preserve the default rate in both sets.

- Train default rate: 41.2%
- Test default rate: 41.5%

WOE encoder fitted on the training set only and applied to both sets. Fitting on the full dataset would constitute data leakage.

### Step 6: Logistic Regression

L2 regularisation. Regularisation parameter C selected via 5-fold cross-validation optimising AUROC across a grid of `[0.001, 0.01, 0.1, 1.0, 10.0]`. Cross-validation selected C=1.0, the weakest regularisation in the grid.

This is itself a diagnostic signal. When cross-validation selects the weakest regularisation available, it means stronger regularisation was removing genuine signal along with noise. In a well-designed experiment with real data, this would indicate features are too weak. Here it additionally reflects the self-referential nature of the target.

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

Train-test Gini gap: 21 points. Industry threshold for concern is 3-4 points.

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

### Surface cause

Weak features with low IV. No feature cleared the strong threshold of 0.30. Cross-validation selected the weakest regularisation in the grid.

### Deeper cause

The target variable was simulated using the same features used in the model. This creates a self-referential relationship. Any predictive signal is a circular reflection of simulation assumptions, not genuine discriminatory power. The Bernoulli noise introduced during simulation diluted even that artificial signal.

### What this means

There are two distinct failure modes in credit risk modelling.

**Implementation failure:** Data leakage, incorrect metric computation, coding errors. Diagnosable from code review.

**Experimental design failure:** Wrong target definition, self-referential feature-target relationships, non-representative sampling. Requires understanding what is actually being measured and whether the measurement is valid.

This project demonstrates the second failure mode. The pipeline is correct. The experiment is not. No amount of regularisation tuning, feature engineering, or binning optimisation could fix a target variable built from the features being modelled.

---

## Key Lessons

**Lesson 1: Interrogate the target variable before anything else**

Before modelling, ask: how was this target defined? Is it genuinely independent of the features? A simulated target built from the feature set will always produce some apparent signal, but that signal is meaningless.

**Lesson 2: IV should be interrogated, not just measured**

Low IV on a self-referential target does not mean the same thing as low IV on a genuine target. IV is only meaningful when the target variable is independent and valid.

**Lesson 3: Cross-validation selecting weak regularisation is a diagnostic**

It points back to data quality or experimental design, not model tuning.

**Lesson 4: Train-test gap diagnosis**

A 21-point Gini gap is overfitting. Here it also reflects the noisiness of a Bernoulli-simulated target -- the model found weak circular patterns in 800 training rows that did not persist in 200 test rows.

**Lesson 5: Pipeline correctness and experimental validity are independent**

A correct pipeline on a flawed experimental design produces meaningless outputs. This is directly relevant to model validation, SR 11-7 compliance, and internal model risk frameworks.

---

## What a Valid Version of This Exercise Would Require

- A dataset with genuine historical default outcomes
- A target variable defined independently of the feature set
- A default rate representative of the actual portfolio (typically 1-5% in retail credit)
- Features with IV above 0.10

With those inputs, this exact pipeline would produce a scorecard with Gini above 0.30 and meaningful score separation between risk groups.

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

Shubham Malpani -- Quantitative Credit Risk Analyst

*This project may contain mistakes. I am open to any feedback that helps my understanding in risk modelling.*
