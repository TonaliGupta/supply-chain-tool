# =========================================
# decision_matrix.py
# Helper functions for Decision Matrix Tool
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------
# 1. Min-Max Normalization Function
# -----------------------------------------
def normalize_min_max(df, col, higher_is_better=True):
    min_val = df[col].min()
    max_val = df[col].max()
    if max_val - min_val == 0:
        return pd.Series([0.5]*len(df))  # avoid divide by zero
    norm = (df[col] - min_val) / (max_val - min_val)
    if not higher_is_better:
        norm = 1 - norm
    return norm

# -----------------------------------------
# 2. Calculate Weighted Scores
# -----------------------------------------
def calculate_weighted_score(df, weights, normalize_cols):
    df_copy = df.copy()
    score = pd.Series(np.zeros(len(df_copy)))
    
    for col, higher_is_better in normalize_cols.items():
        df_copy[col+"_norm"] = normalize_min_max(df_copy, col, higher_is_better)
        score += df_copy[col+"_norm"] * weights[col]
    
    df_copy['WeightedScore'] = score
    return df_copy

# -----------------------------------------
# 3. Monte Carlo Simulation
# -----------------------------------------
def monte_carlo_simulation(df_weighted, weights, normalize_cols, iterations=1000, variation=0.05):
    n_suppliers = len(df_weighted)
    mc_results = np.zeros((iterations, n_suppliers))
    
    for i in range(iterations):
        df_temp = df_weighted.copy()
        for col in normalize_cols.keys():
            df_temp[col + "_sim"] = df_temp[col] * np.random.uniform(1-variation, 1+variation, n_suppliers)
        df_temp_norm = df_temp.copy()
        score = pd.Series(np.zeros(n_suppliers))
        for col, higher_is_better in normalize_cols.items():
            df_temp_norm[col+"_norm"] = normalize_min_max(df_temp, col + "_sim", higher_is_better)
            score += df_temp_norm[col+"_norm"] * weights[col]
        df_temp_norm['MC_Score'] = score
        mc_results[i, :] = df_temp_norm['MC_Score'].values
    
    df_mc = df_weighted.copy()
    df_mc['MC_MeanScore'] = mc_results.mean(axis=0)
    df_mc['MC_Rank'] = df_mc['MC_MeanScore'].rank(ascending=False)
    
    return df_mc, mc_results

# -----------------------------------------
# 4. Recommend Top Suppliers
# -----------------------------------------
def recommend_suppliers(df_mc, top_n=1):
    return df_mc.sort_values('MC_MeanScore', ascending=False).head(top_n)

# -----------------------------------------
# 5. Plot Monte Carlo Boxplot
# -----------------------------------------
def plot_mc_boxplot(df_mc, mc_results):
    plt.figure(figsize=(10,6))
    plt.boxplot(mc_results, labels=df_mc['SupplierID'], showmeans=True)
    plt.xlabel("Supplier")
    plt.ylabel("Monte Carlo Score")
    plt.title("Monte Carlo Score Distribution for Suppliers")
    plt.xticks(rotation=45)
    plt.show()

# -----------------------------------------
# 6. Plot Histogram for Individual Supplier
# -----------------------------------------
def plot_mc_histogram(mc_results, df_mc, supplier_idx=0):
    plt.figure(figsize=(8,5))
    plt.hist(mc_results[:, supplier_idx], bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Monte Carlo Score Distribution for Supplier {df_mc['SupplierID'].iloc[supplier_idx]}")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.show()

