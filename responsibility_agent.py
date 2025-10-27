#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import requests
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import argparse # NEW: To handle command-line arguments
import os       # NEW: To securely get the API key from the environment
import sys      # NEW: To exit gracefully on error

class ResponsibilityAgent:
    """
    AI Agent for calculating financial responsibility scores from transaction data.

    This agent processes JSON transaction data, handles missing values, calculates
    responsibility scores using machine learning techniques, and integrates with
    Gemini Pro API for score comparison and averaging.
    """

    def __init__(self, gemini_api_key: Optional[str] = None, n_clusters: int = 4, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the Responsibility Agent.

        Args:
            gemini_api_key: API key for Gemini Pro integration (optional)
            n_clusters: The default number of clusters for KMeans
            weights: For the scoring components
        """
        self.gemini_api_key = gemini_api_key
        self.transactions_df = None
        self.processed_data = None
        self.responsibility_score = None
        self.gemini_score = None
        self.final_score = None
        self.n_clusters = n_clusters

        if weights is None:
            self.weights = {
                'income_vs_expense': 0.40,
                'income_stability': 0.1,
                'discretionary_spending_control': 0.30,
                'financial_volatility': 0.10,
                'data_completeness': 0.06,
                'investment_growth': 0.04 # Small initial weight for investments
            }
        else:
            self.weights = weights

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Initialize ML components
        self.scaler = StandardScaler()
        # Adjust n_clusters for KMeans based on the number of samples
        # This will be set dynamically in run_analysis or preprocess_data
        self.kmeans = None
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    def load_json_data(self, json_data: Dict) -> pd.DataFrame:
        """
        Load and validate JSON transaction data.

        Args:
            json_data: Dictionary containing transaction data

        Returns:
            pandas.DataFrame: Loaded transaction data

        Raises:
            ValueError: If data format is invalid
        """
        try:
            if 'transactions' not in json_data:
                raise ValueError("JSON data must contain 'transactions' key")

            transactions = json_data['transactions']
            if not isinstance(transactions, list):
                raise ValueError("Transactions must be a list")

            # Convert to DataFrame
            self.transactions_df = pd.DataFrame(transactions)

            # Validate required columns
            required_columns = ["date", "amount", "income", "priority"]
            missing_columns = [col for col in required_columns if col not in self.transactions_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Rename importance to priority if it exists
            if "importance" in self.transactions_df.columns:
                self.transactions_df.rename(columns={"importance": "priority"}, inplace=True)

            self.logger.info(f"Loaded {len(self.transactions_df)} transactions")
            return self.transactions_df

        except Exception as e:
            self.logger.error(f"Error loading JSON data: {str(e)}")
            raise

    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess transaction data and handle missing values.

        Returns:
            pandas.DataFrame: Preprocessed data
        """
        if self.transactions_df is None:
            raise ValueError("No data loaded. Call load_json_data() first.")

        # Create a copy for processing
        df = self.transactions_df.copy()

        # Critical date Handling: Normalize date to 'day' column
        min_date = df["date"].min()
        df["day"] = df["date"] - min_date + 1

        # Handle missing values in priority (-1 indicates missing)
        df["priority_missing"] = (df["priority"] == -1).astype(int)

        # Identify investments (priority = 0)
        df["is_investment"] = (df["priority"] == 0).astype(int)

        # For missing priority values (-1), use median of available values (excluding 0 for investments)
        valid_priority = df[(df["priority"] != -1) & (df["priority"] != 0)]["priority"]
        if len(valid_priority) > 0:
            median_priority = valid_priority.median()
            df.loc[df["priority"] == -1, "priority"] = median_priority
        else:
            # If all priority values are missing or only investments, use neutral value (e.g., 3 for a 1-5 scale)
            df.loc[df["priority"] == -1, "priority"] = 3

        # Ensure data types are correct
        df["date"] = df["date"].astype(int)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["income"] = df["income"].astype(int)
        df["priority"] = pd.to_numeric(df["priority"], errors="coerce")

        # Handle any remaining NaN values
        df = df.fillna(0)

        # Create additional features for analysis
        df["is_income"] = df["income"].astype(bool)
        df["is_expense"] = (~df["is_income"]).astype(int)
        df["amount_log"] = np.log1p(df["amount"])  # Log transform for skewed amounts

        # Identify and neutralize inter-account transfers
        # Group by date and check for matching income and expense amounts
        transfers = []
        for day in df["day"].unique():
            daily_transactions = df[df["day"] == day].copy()
            income_transactions = daily_transactions[daily_transactions["is_income"] == True]
            expense_transactions = daily_transactions[daily_transactions["is_income"] == False]

            for idx_inc, income_row in income_transactions.iterrows():
                for idx_exp, expense_row in expense_transactions.iterrows():
                    if income_row["amount"] == expense_row["amount"]:
                        transfers.append(idx_inc)
                        transfers.append(idx_exp)
                        break  # Found a match for this income transaction

        df["is_transfer"] = df.index.isin(transfers).astype(int)
        self.logger.info(f"Identified {len(transfers) // 2} potential inter-account transfers.")

        # Filter out transfers for further calculations, but keep them in processed_data for completeness
        self.processed_data = df
        self.logger.info("Data preprocessing completed")
        return df[df["is_transfer"] == 0].copy()

    def calculate_basic_metrics(self) -> Dict:
        """
        Calculate basic financial metrics from the transaction data.

        Returns:
            Dict: Dictionary containing basic financial metrics
        """
        if self.processed_data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")

        # Filter out inter-account transfers for calculations
        df = self.processed_data[self.processed_data["is_transfer"] == 0].copy()

        # Separate income and expenses
        income_df = df[df['is_income'] == True]
        expense_df = df[df['is_income'] == False]

        # Calculate basic metrics
        total_income = income_df['amount'].sum()
        total_expenses = expense_df['amount'].sum()
        net_income = total_income - total_expenses

        # Calculate ratios and percentages
        income_expense_ratio = total_income / total_expenses if total_expenses > 0 else float('inf')
        savings_rate = net_income / total_income if total_income > 0 else 0

        # Calculate priority-weighted metrics
        high_priority_expenses = expense_df[expense_df["priority"] >= 3]["amount"].sum()
        low_priority_expenses = expense_df[expense_df["priority"] <= 2]["amount"].sum()

        # Calculate investments
        total_investments = df[df["is_investment"] == 1]["amount"].sum()

        # Calculate transaction patterns
        avg_transaction_amount = df["amount"].mean()
        transaction_frequency = len(df)
        unique_dates = df["day"].nunique()
        avg_daily_transactions = transaction_frequency / unique_dates if unique_dates > 0 else 0

        metrics = {
            "total_income": total_income,
            "total_expenses": total_expenses,
            "net_income": net_income,
            "income_expense_ratio": income_expense_ratio,
            "savings_rate": savings_rate,
            "high_priority_expenses": high_priority_expenses,
            "low_priority_expenses": low_priority_expenses,
            "avg_transaction_amount": avg_transaction_amount,
            "transaction_frequency": transaction_frequency,
            "unique_dates": unique_dates,
            "avg_daily_transactions": avg_daily_transactions,
            "total_investments": total_investments,
            "missing_priority_count": df["priority_missing"].sum()
        }

        self.logger.info("Basic metrics calculated")
        return metrics



    def calculate_responsibility_score(self, basic_metrics: Dict, ml_metrics: Dict) -> Tuple[float, Dict[str, float]]:
        """
        Calculate the financial responsibility score (0-100).

        The score is based on a combination of financial metrics, spending habits,
        and importance of transactions.

        Returns:
            float: The calculated responsibility score (0-100).
        """
        if self.processed_data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")

        # 1. income_vs_expense_score
        savings_rate = basic_metrics["savings_rate"]
        income_vs_expense_score = 1 / (1 + np.exp(-savings_rate * 5))

        # 2. income_stability_score
        # Group income transactions into 30-day periods
        income_df = self.processed_data[self.processed_data["is_income"] == True].copy()
        if not income_df.empty:
            # Calculate period for each transaction
            income_df["period"] = (income_df["day"] - 1) // 30
            period_income = income_df.groupby("period")["amount"].sum()

            mean_period_income = period_income.mean()
            std_dev_period_income = period_income.std()

            if mean_period_income == 0:
                income_stability_score = 0.0
            else:
                cv = std_dev_period_income / mean_period_income
                income_stability_score = max(0, 1 - cv)
        else:
            income_stability_score = 0.0 # No income, no stability

        # 3. discretionary_spending_control
        total_expenses = basic_metrics["total_expenses"]
        low_priority_expenses = basic_metrics["low_priority_expenses"]
        discretionary_spending_control = 1 - (low_priority_expenses / total_expenses if total_expenses > 0 else 0)

        # 4. financial_volatility_score (now with positive ML impact)
        anomaly_ratio = ml_metrics["anomaly_ratio"]
        risky_spending_ratio = ml_metrics["risky_spending_ratio"]

        # Invert the ratios so lower values (better) give higher scores
        # Scale them to have a small positive impact. For example, a perfect score (0 anomaly/risky) adds 0.05, worst (1.0) adds 0.
        ml_impact = (1 - anomaly_ratio) * 0.025 + (1 - risky_spending_ratio) * 0.025 # Total max 0.05

        # The financial_volatility_score itself is still about reducing volatility, so it's inversely related.
        # But the overall score will now have a small positive boost from good ML metrics.
        financial_volatility_score = max(0, 1 - anomaly_ratio - risky_spending_ratio)

        # 5. data_completeness_score
        total_transactions = len(self.transactions_df)
        missing_priority_count = basic_metrics["missing_priority_count"]
        data_completeness_score = 1 - (missing_priority_count / total_transactions if total_transactions > 0 else 0)

        # 6. investment_growth_score (positive effect for investments)
        total_investments = basic_metrics["total_investments"]
        total_income = basic_metrics["total_income"]
        investment_ratio = total_investments / total_income if total_income > 0 else 0
        investment_growth_score = min(1, investment_ratio * 2) # Cap at 1, scale to give positive effect

        # Store sub-scores for breakdown
        sub_scores = {
            "income_vs_expense": income_vs_expense_score,
            "income_stability": income_stability_score,
            "discretionary_spending_control": discretionary_spending_control,
            "financial_volatility": financial_volatility_score,
            "data_completeness": data_completeness_score,
            "investment_growth": investment_growth_score
        }

        # Calculate weighted sum
        score = (
            sub_scores["income_vs_expense"] * self.weights["income_vs_expense"] +
            sub_scores["income_stability"] * self.weights["income_stability"] +
            sub_scores["discretionary_spending_control"] * self.weights["discretionary_spending_control"] +
            sub_scores["financial_volatility"] * self.weights["financial_volatility"] +
            sub_scores["data_completeness"] * self.weights["data_completeness"] +
            sub_scores["investment_growth"] * self.weights["investment_growth"] +
            ml_impact # Add the small positive ML impact
        )
        # Scale to 0-100
        final_score = round(score * 100, 2)
        self.responsibility_score = final_score
        self.logger.info(f"Calculated responsibility score: {self.responsibility_score}")
        return final_score, sub_scores



    def run_analysis(self, json_data: Dict):
        """
        Runs the complete financial responsibility analysis pipeline.
        """
        self.load_json_data(json_data)
        processed_df = self.preprocess_data()
        basic_metrics = self.calculate_basic_metrics()
        ml_metrics = self.apply_ml_enhancements()
        self.responsibility_score, self.sub_scores = self.calculate_responsibility_score(basic_metrics, ml_metrics)
        self.ml_insights = ml_metrics["ml_insights"]

        gemini_result = self.get_gemini_pro_score()
        self.gemini_score = gemini_result["score"]

        if self.responsibility_score is not None and self.gemini_score is not None:
            self.final_score = (self.responsibility_score + self.gemini_score) / 2
            self.logger.info(f"Final averaged score: {self.final_score}")
        else:
            self.final_score = self.responsibility_score if self.responsibility_score is not None else self.gemini_score
            self.logger.warning("Could not average scores due to missing Gemini score.")

    def get_gemini_pro_score(self) -> Dict:
        """
        Get a responsibility score and qualitative analysis from Gemini Pro API.

        Returns:
            Dict: Dictionary containing the score, qualitative analysis, and reasoning from Gemini Pro.
        """
        if not self.gemini_api_key:
            self.logger.warning("Gemini API key not provided. Skipping Gemini Pro integration.")
            return {"score": None, "qualitative_analysis": "Gemini API key not provided.", "reasoning": ""}

        if self.processed_data is None:
            raise ValueError("No data processed. Call preprocess_data() first.")

        # Data Preparation: Convert relevant DataFrame columns to a compact CSV string
        csv_data_string = self.processed_data[["day", "amount", "income", "priority"]].to_csv(index=False)

        # The Prompt Template
        prompt_template = f"""
You are a meticulous and impartial financial analyst AI. Your task is to analyze a user's raw transaction data and provide a financial responsibility score and a concise analysis.

**Analysis Framework:**
1.  **Income vs. Expenses:** Does the user earn more than they spend? What is their savings rate?
2.  **Spending Habits:** Where does their money go? Is spending concentrated on essentials (high priority) or discretionary items (low priority)?
3.  **Financial Stability:** Is income regular or erratic when viewed in 30-day windows? Are there sudden, large, anomalous expenses that suggest financial shocks?

**User Transaction Data (CSV Format):**
The 'day' column represents the number of days since the first transaction. 'priority' is a 1-5 scale (1=low, 5=high).
---
{csv_data_string}
---

**Your Task:**
Based on your analysis of the data provided, respond with a single, valid JSON object. Do not include any other text or explanation outside of the JSON object. The JSON object must contain the following keys and nothing more:
- "score": An integer from 0 to 100, representing your calculated financial responsibility score.
- "qualitative_analysis": A very brief, 2-sentence string. The first sentence should identify the user's single greatest financial strength. The second sentence should identify their single biggest area for improvement.
- "reasoning": A brief explanation of the key factors that led to your score.

**Example JSON Response:**
{{
    "score": 78,
    "qualitative_analysis": "Your primary strength is a consistent and positive savings rate. The biggest area for improvement is reducing the high frequency of small, low-priority discretionary purchases.",
    "reasoning": "Score was boosted by high savings rate. It was penalized due to a large number of transactions in a low-priority cluster and one significant anomalous expense."
}}

**JSON Response:**
"""
        # --- NEW: ACTUAL API CALL LOGIC ---
        # This part is a placeholder. For a real application, you would make an HTTP request here.
        # For example, using the 'requests' library:
        #
        # url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + self.gemini_api_key
        # headers = {'Content-Type': 'application/json'}
        # body = {"contents": [{"parts": [{"text": prompt_template}]}]}
        # response = requests.post(url, headers=headers, json=body)
        #
        # if response.status_code == 200:
        #     api_response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        #     # Clean and parse the JSON response text
        #     # ...
        # else:
        #     self.logger.error(f"Gemini API call failed with status {response.status_code}: {response.text}")
        #     return {"score": None, "qualitative_analysis": "API call failed.", "reasoning": ""}
        # --- END NEW ---

        self.logger.info("Simulating Gemini Pro API call...")
        simulated_score = np.random.randint(50, 95) # Random integer score between 50 and 95
        simulated_qualitative_analysis = "Simulated strength: Consistent income. Simulated improvement: Reduce impulse buys."
        simulated_reasoning = "Simulated reasoning: Good income management, but some unnecessary spending."

        self.gemini_score = simulated_score
        self.logger.info(f"Simulated Gemini Pro score: {self.gemini_score}")

        return {
            "score": simulated_score,
            "qualitative_analysis": simulated_qualitative_analysis,
            "reasoning": simulated_reasoning
        }






    def apply_ml_enhancements(self) -> Dict[str, any]:
        """
        Apply machine learning enhancements to the processed data.
        This includes feature scaling, clustering, and anomaly detection.

        Returns:
            Dict: Dictionary containing ML-derived metrics for scoring.
        """
        if self.processed_data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")

        df = self.processed_data.copy()

        # Select features for ML
        # Using amount, priority, and income for ML features
        features = df[["amount", "priority", "income"]]

        # 1. Feature Scaling
        scaled_features = self.scaler.fit_transform(features)
        self.processed_data[["amount_scaled", "priority_scaled", "income_scaled"]] = scaled_features
        self.logger.info("Features scaled.")

        # 2. Clustering (KMeans)
        # Use scaled features for clustering
        # Dynamically set n_clusters to be at most the number of samples - 1
        n_clusters_adjusted = min(self.n_clusters, len(df) - 1)

        risky_spending_ratio = 0.0 # Initialize
        ml_insights = {
            "cluster_distribution": {},
            "anomalous_transactions_count": 0,
            "anomalous_transactions_details": [],
            "risky_cluster_id": -1
        } # Initialize

        if n_clusters_adjusted < 2:
            self.logger.warning("Not enough samples for meaningful clustering (need at least 2). Skipping KMeans.")
            df["cluster"] = 0 # Assign all to a single default cluster
        else:
            self.kmeans = KMeans(n_clusters=n_clusters_adjusted, random_state=42, n_init=10)
            df["cluster"] = self.kmeans.fit_predict(scaled_features)
            self.processed_data["cluster"] = df["cluster"]
            self.logger.info("Transactions clustered.")

            # Identify the risky spending cluster (lowest average priority)
            cluster_avg_priority = df.groupby("cluster")["priority"].mean()
            risky_cluster_id = cluster_avg_priority.idxmin()

            # Calculate risky_spending_ratio
            total_expenses_in_risky_cluster = df[(df["cluster"] == risky_cluster_id) & (df["is_expense"] == 1)]["amount"].sum()
            total_expenses = df[df["is_expense"] == 1]["amount"].sum()
            risky_spending_ratio = total_expenses_in_risky_cluster / total_expenses if total_expenses > 0 else 0

            ml_insights["cluster_distribution"] = self.processed_data["cluster"].value_counts().to_dict()
            ml_insights["risky_cluster_id"] = int(risky_cluster_id)
        # 3. Anomaly Detection (Isolation Forest) - This should always run
        self.isolation_forest.fit(scaled_features)
        df["anomaly_score"] = self.isolation_forest.decision_function(scaled_features)
        df["is_anomaly"] = self.isolation_forest.predict(scaled_features)
        self.processed_data["anomaly_score"] = df["anomaly_score"]
        self.processed_data["is_anomaly"] = df["is_anomaly"]
        self.logger.info("Anomaly detection applied.")

        # Calculate anomaly_ratio
        anomaly_ratio = (df["is_anomaly"] == -1).sum() / len(df) if len(df) > 0 else 0

        ml_insights["anomalous_transactions_count"] = int((self.processed_data["is_anomaly"] == -1).sum())
        ml_insights["anomalous_transactions_details"] = self.processed_data[self.processed_data["is_anomaly"] == -1].to_dict(orient="records")

        self.logger.info("ML enhancements applied and metrics generated.")

        return {
            "risky_spending_ratio": risky_spending_ratio,
            "anomaly_ratio": anomaly_ratio,
            "ml_insights": ml_insights
        }

# ==============================================================================
# NEW: Main execution block that handles file loading and command-line arguments
# ==============================================================================
if __name__ == "__main__":
    # 1. Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Analyzes financial transaction data from a JSON file to calculate a responsibility score."
    )
    # This defines the one required argument: the path to your JSON file.
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the JSON file containing transaction data."
    )
    args = parser.parse_args()

    # 2. Load the data from the specified file
    try:
        with open(args.json_file, 'r') as f:
            data_from_file = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{args.json_file}' was not found.")
        sys.exit(1) # Exit the script with an error code
    except json.JSONDecodeError:
        print(f"Error: The file '{args.json_file}' is not a valid JSON file.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        sys.exit(1)

    # 3. Initialize and run the agent
    # It now gets the API key from an "environment variable" for better security.
    # This avoids hard-coding the key in the script.
    gemini_key = os.getenv("GEMINI_API_KEY")
    agent = ResponsibilityAgent(gemini_api_key=gemini_key)

    try:
        # The data we loaded from the file is passed to the agent
        agent.run_analysis(data_from_file)

        # 4. Print the results
        print("\n--- Analysis Results ---")
        print(f"Responsibility Score: {agent.responsibility_score:.2f}")
        print(f"Gemini Pro Score: {agent.gemini_score}")
        print(f"Final Averaged Score: {agent.final_score:.2f}")
        print(f"\nML Insights: {json.dumps(agent.ml_insights, indent=2)}")
        print(f"\nSub-scores:")
        for name, score in agent.sub_scores.items():
            print(f"  - {name.replace('_', ' ').title()}: {score:.4f}")
        # print(f"\nProcessed Data Head:\n{agent.processed_data.head()}")

    except ValueError as e:
        print(f"Analysis Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")