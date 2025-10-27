# Finsight_AI_engine
# ResponsibilityAgent

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/your-repo/responsibility-agent)

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

The ResponsibilityAgent is an AI-powered financial analysis system that evaluates personal financial responsibility by analyzing transaction data. It combines traditional financial metrics with machine learning to generate a comprehensive responsibility score from 0 to 100.

## Core Features

-   **Data-Driven Assessment**: Analyzes actual transaction patterns, not self-reported data.
-   **Multi-Dimensional Scoring**: Considers income, spending, volatility, and investment behavior.
-   **Machine Learning Insights**: Uses K-Means clustering and Isolation Forest anomaly detection to uncover deeper patterns.
-   **Configurable Weights**: Allows customization of the scoring algorithm to fit different financial philosophies.
-   **Comprehensive Analysis**: Provides a final score, detailed component scores, and insights into financial habits.

## How It Works

The agent follows a robust pipeline to generate its analysis:

1.  **Load & Validate Data**: Ingests JSON transaction data and validates its structure.
2.  **Preprocess & Clean**: Normalizes dates, handles missing values, and identifies transfers.
3.  **Calculate Basic Metrics**: Computes key indicators like savings rate, spending ratios, and income stability.
4.  **Apply ML Enhancements**: Identifies risky spending clusters and anomalous transactions.
5.  **Calculate Responsibility Score**: Combines all metrics using a weighted algorithm.
6.  **Generate Final Report**: Averages the internal score with an external AI validation (simulated) for a final, robust assessment.

## Scoring Algorithm

The final score is a weighted average of six key dimensions of financial responsibility:

| Component                        | Default Weight | What it Measures                                        |
| -------------------------------- | :------------: | ------------------------------------------------------- |
| **Income vs. Expense**           |      40%       | The ability to live within one's means (savings rate).  |
| **Discretionary Spending Control** |      30%       | Discipline in prioritizing essential vs. non-essential spending. |
| **Income Stability**             |      10%       | The consistency and predictability of income streams.     |
| **Financial Volatility**         |      10%       | The prevalence of anomalous transactions and risky spending. |
| **Data Completeness**            |       6%       | The quality and completeness of provided transaction data. |
| **Investment Growth**            |       4%       | Positive reinforcement for investment and asset growth.   |
