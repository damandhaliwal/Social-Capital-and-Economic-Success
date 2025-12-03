# Quantile Regression with entire dataset
# Daman Dhaliwal

# import libraries
import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_prep import merged_combined
from utils import paths

def run_quantreg(overwrite = False):
    df = merged_combined(overwrite = overwrite)
    df = df.to_pandas()

    data_2019 = df[df['file_year'] == 2019][['abi', 'fips', 'sales', 'employees', 'naics', 'ec', 'clustering', 'civic']].copy()
    data_2019 = data_2019.rename(columns={'sales': 'sales_2019', 'employees': 'emp_2019'})

    data_2024 = df[df['file_year'] == 2024][['abi', 'sales']].copy()
    data_2024 = data_2024.rename(columns={'sales': 'sales_2024'})

    merged = data_2019.merge(data_2024, on='abi', how='left')
    merged['sales_2024'] = merged['sales_2024'].fillna(0)

    # compute sales change
    merged['survived'] = (merged['sales_2024'] > 0).astype(int)

    merged = merged.dropna(subset = ['sales_2019', 'ec', 'fips', 'emp_2019', 'naics', 'clustering', 'civic'])
    merged = merged[merged['sales_2019'] > 0]

    # filter for survivors only (intensive margin)
    merged = merged[merged['sales_2024'] > 0]

    # log sales
    merged['log_sales_2019'] = np.log1p(merged['sales_2019'])
    merged['log_sales_2024'] = np.log1p(merged['sales_2024'])
    merged['log_sales_change'] = merged['log_sales_2024'] - merged['log_sales_2019']

    # standardize independent variables
    merged['ec_std'] = (merged['ec'] - merged['ec'].mean()) / merged['ec'].std()
    merged['clustering_std'] = (merged['clustering'] - merged['clustering'].mean()) / merged['clustering'].std()
    merged['civic_std'] = (merged['civic'] - merged['civic'].mean()) / merged['civic'].std()
    merged['log_emp_2019'] = np.log1p(merged['emp_2019'])
    merged['naics2'] = merged['naics'].str[:2]

    # define quantiles from 0.05 to 0.95
    quantiles = np.arange(0.05, 1.00, 0.05)

    results = []

    for q in tqdm(quantiles, desc="   Quantiles", ncols=70):
        X = merged[['ec_std', 'clustering_std', 'civic_std', 'log_sales_2019']].assign(const=1)
        model = QuantReg(merged['log_sales_change'], X)
        res = model.fit(q=q)

        results.append({
            'quantile': q,
            'ec_coef': res.params['ec_std'],
            'ec_se': res.bse['ec_std'],
            'ec_pval': res.pvalues['ec_std'],
            'clustering_coef': res.params['clustering_std'],
            'clustering_se': res.bse['clustering_std'],
            'clustering_pval': res.pvalues['clustering_std'],
            'civic_coef': res.params['civic_std'],
            'civic_se': res.bse['civic_std'],
            'civic_pval': res.pvalues['civic_std']
        })

    results_df = pd.DataFrame(results)

    print("\nEconomic Connectedness:")
    for _, row in results_df.iterrows():
        sig = '***' if row['ec_pval'] < 0.01 else '**' if row['ec_pval'] < 0.05 else '*' if row['ec_pval'] < 0.1 else ''
        print(f"   {row['quantile']:<10.2f} {row['ec_coef']:>12.4f} {row['ec_se']:>12.4f} {row['ec_pval']:>10.4f} {sig:>5}")

    print("\nClustering:")
    for _, row in results_df.iterrows():
        sig = '***' if row['clustering_pval'] < 0.01 else '**' if row['clustering_pval'] < 0.05 else '*' if row['clustering_pval'] < 0.1 else ''
        print(f"   {row['quantile']:<10.2f} {row['clustering_coef']:>12.4f} {row['clustering_se']:>12.4f} {row['clustering_pval']:>10.4f} {sig:>5}")

    print("\nCivic Engagement:")
    for _, row in results_df.iterrows():
        sig = '***' if row['civic_pval'] < 0.01 else '**' if row['civic_pval'] < 0.05 else '*' if row['civic_pval'] < 0.1 else ''
        print(f"   {row['quantile']:<10.2f} {row['civic_coef']:>12.4f} {row['civic_se']:>12.4f} {row['civic_pval']:>10.4f} {sig:>5}")

    # save results
    path = paths()['data']
    filename = os.path.join(path, 'quant_reg_results.csv')
    results_df.to_csv(filename, index=False)

    return results_df, merged


def plot_quantile_results(results_df):

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    # Variables to plot mapping
    variables = [
        {'col': 'ec', 'title': 'Economic Connectedness', 'color': '#1f77b4'},
        {'col': 'clustering', 'title': 'Cohesion (Clustering)', 'color': '#d62728'},
        {'col': 'civic', 'title': 'Civic Engagement', 'color': '#2ca02c'} 
    ]
    
    for i, var in enumerate(variables):
        ax = axes[i]
        col_name = var['col']
        
        quantiles = results_df['quantile']
        coefs = results_df[f'{col_name}_coef']
        errors = results_df[f'{col_name}_se']
        
        upper = coefs + 1.96 * errors
        lower = coefs - 1.96 * errors
        
        # Plot coefficients
        ax.plot(quantiles, coefs, color=var['color'], lw=2, label='Quantile Estimate')
        
        ax.fill_between(quantiles, lower, upper, color=var['color'], alpha=0.2, label='95% CI')
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        
        ax.set_title(var['title'], fontsize=14, fontweight='bold')
        ax.set_xlabel('Quantile ($tau$)', fontsize=12)
        if i == 0:
            ax.set_ylabel('Coefficient Estimate (Log Sales Change)', fontsize=12)
        
        ax.set_xticks(np.arange(0.1, 1.0, 0.1))
        
    plt.tight_layout()
    path = paths()['figures']
    filename = os.path.join(path, 'quantile_regression_results.png')
    fig.savefig(filename, dpi = 600)

if __name__ == "__main__":
    results, data = run_quantreg() 

    plot_quantile_results(results)
