# Social Capital and Economic Success: A Causal Inference Analysis

This repository contains the codebase for an empirical analysis investigating the causal impact of social capital on business resilience and growth during the COVID-19 pandemic.

Leveraging high-dimensional administrative data (\~14 million firms) and the Social Capital Atlas, this project applies **Double/Debiased Machine Learning (DML)** and **Quantile Regression** to isolate heterogeneous treatment effects across industries and firm performance distributions.

## 🚀 Project Overview

  * **Objective:** Decompose the effects of Economic Connectedness (EC), Social Cohesion, and Civic Engagement on firm survival (extensive margin) and revenue growth (intensive margin) following an exogenous economic shock.
  * **Scale:** Analyzed a panel of **14 million+ business establishments** merged with county-level social graph data.
  * **Key Findings:** \* **Social Cohesion** acts as a safety net, significantly increasing survival probability during crises (approx. 1pp increase per SD).
      * **Economic Connectedness** exhibits non-linear effects: it amplifies growth for high-performing firms (90th percentile) but negatively impacts struggling firms.
      * **Heterogeneity:** Positive effects are concentrated in information-heavy sectors (e.g., Tech, Logistics), with negative effects observed in independent healthcare practices during the pandemic.

## 🛠 Tech Stack & Methodology

The pipeline is implemented in **Python** with a focus on efficient data processing and robust causal estimation.

### Core Libraries

  * **Causal ML:** `DoubleML`, `DoWhy`
  * **Machine Learning:** `XGBoost`, `Scikit-Learn`
  * **Data Engineering:** `Polars` (for high-performance ETL), `Pandas`
  * **Statistical Analysis:** `Statsmodels` (Quantile Regression, OLS)

### Methodological Approach

#### 1\. Double/Debiased Machine Learning (DML)

To address omitted variable bias and regularization bias inherent in high-dimensional controls, I implemented **Partially Linear Regression (PLR)** models using the `DoubleML` framework.

  * **Nuisance Estimation:** Utilized **XGBoost** (Gradient Boosted Trees) to flexibly model the relationship between confounders ($X$), treatment ($T$), and outcome ($Y$).
      * *Outcome Model:* $g(X) = E[Y|X]$
      * *Treatment Model:* $m(X) = E[T|X]$
  * **Orthogonalization:** Regressed residualized outcomes ($Y - g(X)$) on residualized treatments ($T - m(X)$) to obtain valid causal estimates ($\theta$).
  * **Cross-Fitting:** Applied 5-fold cross-fitting to prevent overfitting and ensure valid inference.

#### 2\. Heterogeneity Analysis

  * **Industry-Specific Estimators:** Extended the DML framework to estimate Group Average Treatment Effects (GATEs) for 2-digit and 4-digit NAICS codes.
  * **Quantile Regression:** Estimated conditional quantiles ($\tau \in [0.05, 0.95]$) to analyze how social capital effects vary across the distribution of firm growth, testing hypotheses on "safety nets" vs. "amplifiers."

#### 3\. Data Pipeline

  * **ETL:** Built a robust pipeline using `Polars` to process 5 years of raw business data (GBs of .txt files), normalize schemas, and perform fuzzy merging with social capital indices.
  * **Robustness:** Implemented placebo tests and refutation methods using `DoWhy` to test sensitivity to unobserved confounders.

## 📂 Repository Structure

```bash
├── Code/
│   ├── data_prep.py          # ETL pipeline using Polars for cleaning and merging datasets
│   ├── dml.py                # Double Machine Learning implementation (XGBoost + DoubleML)
│   ├── dml_sub_industry.py   # Heterogeneity analysis at the 4-digit NAICS level
│   ├── quantreg.py           # Quantile regression for distributional effects
│   ├── ols.py                # Baseline OLS specifications with fixed effects
│   ├── dowhy.py              # Causal refutation and robustness checks
│   ├── data_description.py   # Summary statistics and placebo tests
│   └── utils.py              # Path management and utility functions
├── Text/                     # Latex source for the associated research paper
├── Output/                   # Generated models, tables, and plots
└── README.md
```

## 📊 Key Results Summary

| Metric | Methodology | Finding |
| :--- | :--- | :--- |
| **Firm Survival** | **Double ML (XGB)** | Social Cohesion is the primary driver of survival ($\beta \approx 0.01^{***}$), outperforming Economic Connectedness. |
| **Sales Growth** | **Quantile Reg** | Economic Connectedness is detrimental at lower quantiles ($\tau < 0.2$) but highly beneficial at upper quantiles ($\tau > 0.8$). |
| **Sector Impact** | **Heterogeneity** | Transport & Information sectors see highest ROI on social capital; Healthcare sees negative ROI during pandemic conditions. |

## 💻 Usage

To reproduce the analysis:

1.  **Environment Setup:**

    ```bash
    pip install polars pandas doubleml xgboost statsmodels dowhy
    ```

2.  **Data Generation:**
    Build the panel dataset from raw sources (requires source files in `Data/Input`).

    ```bash
    python Code/data_prep.py
    ```

3.  **Run Causal Estimators:**
    Execute the Double ML and Quantile Regression pipelines.

    ```bash
    python Code/dml.py
    python Code/quantreg.py
    ```

-----

*Author: Damanveer Singh Dhaliwal*
