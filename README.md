# Superstore Analysis & Predictive Modeling

This project analyzes U.S. “Superstore” sales data to identify profitability drivers and predict loss-making transactions. The workflow includes **Exploratory Data Analysis (EDA)**, **Unsupervised Learning (Clustering)**, and **Supervised Learning (Classification)**.

## 📂 Project Structure

```text
.
├── code/
│   ├── setup_and_eda.R    # Data cleaning and descriptive analysis
│   ├── supervised.R       # Predictive models (XGBoost, RF, etc.)
│   └── unsupervised.R     # PCA and clustering (K-means, Hierarchical)
├── renv/                  # Project dependency management
├── .gitignore             # Excludes unnecessary files (e.g., local CSVs)
├── .Rprofile              # Local R configuration
├── renv.lock              # Snapshot of packages for reproducibility
├── superstore.Rproj       # RStudio project file
└── Report_Finale.pdf      # Full documentation (end-to-end analysis)
```

---

## 🛠️ How to Reproduce the Project

This project uses `renv` to ensure portability and consistent R package versions.

1. **Clone the repository**

```bash
git clone https://github.com/your-username/superstore-analysis.git
```

2. **Open the project**
   Double-click `superstore.Rproj` in RStudio.

3. **Restore the environment**
   If RStudio doesn’t prompt you automatically, run:

```r
renv::restore()
```

4. **Data**
   The dataset was found in Kaggle as "superstore" (different versions), you can download locally and update the path to `SampleSuperstore.csv` in `setup_and_eda.R` (variable `file_path`).

---

## 🧩 Module Overview

### 1. `setup_and_eda.R`

Handles dataset loading and cleaning (e.g., standardizing column names, removing redundant fields). Includes extensive visual EDA:

* Correlation analysis across numeric variables
* Target distribution for `loss_flag`
* Geographic analysis (U.S. maps) of profit by region/state
* Discount impact on profitability

### 2. `unsupervised.R`

Advanced exploratory analysis for segmentation:

* **PCA:** Dimensionality reduction to visualize variance structure
* **Clustering:** **K-means** and **Hierarchical Clustering** (Ward’s method)
* **Profiling:** Cluster-level summaries to define business segments

### 3. `supervised.R`

Builds models to predict whether a transaction will generate a loss (`loss_flag`):

* **Models:** Logistic Regression, LASSO, Ridge, Random Forest, GBM, XGBoost
* **Evaluation:** Precision, Recall, F1-score, and AUC comparison
* **Interpretability:** Feature importance and Partial Dependence Plots (PDP) to isolate the effect of discount on loss probability

---

## 📄 Final Report

`final_report.pdf` provides the strategic summary of the entire study, translating technical findings into business insights. It includes conclusions on the best-performing models and actionable recommendations to optimize the Superstore’s discount strategy.

---