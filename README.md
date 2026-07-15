# Superstore Analysis & Predictive Modeling

Analysis of U.S. "Superstore" retail sales data to identify profitability drivers and predict loss-making transactions.

The project is written entirely in **R** and covers three stages: exploratory data analysis (EDA), unsupervised learning (PCA and clustering), and supervised learning (binary classification on `loss_flag`).

## Dataset

The data comes from the public **[Superstore](https://www.kaggle.com/datasets/ibrahimelsayed182/superstore/data)** dataset on Kaggle (~10k U.S. retail transactions: sales, quantity, discount, and profit, along with geographic and product dimensions).

The CSV file is not included in this repository. To run the project:

1. Download the dataset from Kaggle
2. Save the file locally as `SampleSuperstore.csv`
3. Update the `file_path` variable in `code/setup_and_eda.R`

## Project structure

```text
.
├── code/
│   ├── setup_and_eda.R    # Data loading, cleaning, and descriptive analysis
│   ├── unsupervised.R     # PCA and clustering (K-means, hierarchical)
│   └── supervised.R       # Predictive models (XGBoost, RF, GLM, ...)
├── renv/                  # Dependency management
├── renv.lock              # Package snapshot for reproducibility
├── superstore.Rproj       # RStudio project file
├── final_report.pdf       # Full analysis report
├── .Rprofile
└── .gitignore
```

## Modules

### `setup_and_eda.R`

Loads and cleans the dataset (standardizing column names, dropping redundant fields) and builds the `loss_flag` target. Includes:

- Correlation analysis across numeric variables
- Target distribution
- Geographic analysis of profit by region and state (U.S. maps)
- Impact of discount on profitability

### `unsupervised.R`

Transaction segmentation:

- **PCA** for dimensionality reduction and reading the variance structure
- **Clustering** with K-means and hierarchical methods (Ward's linkage)
- **Profiling** of clusters to define business segments

### `supervised.R`

Classification of loss-making transactions:

- **Models:** logistic regression, LASSO, Ridge, Random Forest, GBM, XGBoost
- **Evaluation:** precision, recall, F1-score, and AUC comparison
- **Interpretability:** feature importance and Partial Dependence Plots to isolate the effect of discount on loss probability

## Final report

[`final_report.pdf`](final_report.pdf) contains the full write-up: methodology, model comparison, results, and actionable recommendations for optimizing the Superstore's discount strategy.
