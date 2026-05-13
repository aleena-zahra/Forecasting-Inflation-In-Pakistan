# Forecasting Inflation in Developing Economies: A Machine Learning Approach Applied to Pakistan

## Citation 
Developed as an undergraduate research project by Aleena Zahra and Muaz Abdullah, June 2025.
This work is the authors' own independent research. Not to be reproduced without citation. 

Please cite as Zahra, A., & Abdullah, M. M. (2025). A Data-Driven Approach to Inflation Forecasting in Pakistan: Comparative Analysis of ARIMA, SVM, Tree-Based Models, and Regularized Regression(1989-2024) (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.15699881

Updated version on SSRN as a preprint https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5355948

## Overview

This project employs advanced machine learning and statistical techniques to forecast inflation rates in Pakistan. By utilizing a variety of models—ARIMA, Ridge, Lasso, Elastic Net, Support Vector Regression, and Decision Trees—the project offers a comprehensive look at inflation dynamics and predictive performance.

## Project Significance

- **Policy Relevance:** Reliable inflation forecasts aid government agencies and the central bank in designing economic policies.
- **Comparative Modelling:** Demonstrates the strengths and weaknesses of different modeling approaches, helping researchers choose the optimal framework.
- **Economic Insights:** Provides stakeholders with critical information for business planning, investment strategies, and economic research.

## Methods Used

- **ARIMA:** For time series modeling and capturing autocorrelation in inflation data.
- **Ridge & Lasso Regression:** To handle feature selection and mitigate overfitting.
- **Elastic Net:** A compromise between Ridge and Lasso.
- **Support Vector Regression (SVR):** For nonlinear regression tasks.
- **Decision Trees:** For capturing complex, nonlinear relationships in data.

All analyses are performed using R.

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/aleena-zahra/Forecasting-Inflation-In-Pakistan.git
    ```
2. **Open the R scripts or RMarkdown files in RStudio or your preferred IDE.**

3. **Install required R packages (e.g., forecast, glmnet, e1071, rpart).**

4. **Run the code** to reproduce the models and visualizations.

## Directory Structure

- `data/` — Includes raw and processed datasets.
- `scripts/` — Contains code for each modeling technique.
- `results/` — Output graphs, tables, and summary statistics.

## Contributing

Ideas and suggestions for improving the analysis or adding new models are highly encouraged.

## License

Distributed under the MIT License. See `LICENSE` for details.
