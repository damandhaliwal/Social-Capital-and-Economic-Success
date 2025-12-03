# Data Description
# Daman Dhaliwal

# import libraries
import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf

from utils import paths
from data_prep import merged_combined, merged_survival

# output summary statistics
def summary_stats(overwrite=False):
    data = merged_survival(overwrite=overwrite).to_pandas()
    
    required_cols = ['survived_2024', 'ec', 'employees', 'sales', 'fips', 'naics', 'naics2']
    data = data.dropna(subset=required_cols)
    
    # For growth sample (survivors only) - keep separate
    df = merged_combined(overwrite=overwrite).to_pandas()
    
    data_2019 = df[df['file_year'] == 2019][['abi', 'sales']].copy()
    data_2019 = data_2019.rename(columns={'sales': 'sales_2019'})
    
    data_2024 = df[df['file_year'] == 2024][['abi', 'sales']].copy()
    data_2024 = data_2024.rename(columns={'sales': 'sales_2024'})
    
    growth = data_2019.merge(data_2024, on='abi', how='inner')
    growth = growth[growth['sales_2019'] > 0]
    growth = growth[growth['sales_2024'] > 0]
    growth['log_sales_change'] = np.log1p(growth['sales_2024']) - np.log1p(growth['sales_2019'])

    variables = ['survived_2024', 'sales', 'employees', 'ec', 'clustering', 'civic']
    
    stats = data[variables].agg(['count', 'mean', 'std', 'min', 'median', 'max']).T
    stats.columns = ['N', 'Mean', 'SD', 'Min', 'Median', 'Max']
    stats['N'] = stats['N'].astype(int)
    
    # Add growth row
    growth_row = pd.DataFrame({
        'N': [len(growth)],
        'Mean': [growth['log_sales_change'].mean()],
        'SD': [growth['log_sales_change'].std()],
        'Min': [growth['log_sales_change'].min()],
        'Median': [growth['log_sales_change'].median()],
        'Max': [growth['log_sales_change'].max()],
    }, index=['log_sales_change'])
    
    stats = pd.concat([stats.iloc[:3], growth_row, stats.iloc[3:]])
    
    print(stats.round(3).to_string())
    
    return stats

def run_placebo(overwrite=False):
    df = merged_combined(overwrite=overwrite).to_pandas()
    
    # Baseline: 2016
    data_2016 = df[df['file_year'] == 2016][['abi', 'fips', 'sales', 'employees', 'naics', 'ec', 'clustering', 'civic']].copy()
    data_2016 = data_2016.rename(columns={'sales': 'sales_2016', 'employees': 'emp_2016'})
    
    # Outcome: 2019
    data_2019 = df[df['file_year'] == 2019][['abi', 'sales']].copy()
    data_2019 = data_2019.rename(columns={'sales': 'sales_2019'})
    
    merged = data_2016.merge(data_2019, on='abi', how='left')
    merged['sales_2019'] = merged['sales_2019'].fillna(0)
    
    merged = merged.dropna(subset=['sales_2016', 'ec', 'fips', 'emp_2016', 'naics', 'clustering', 'civic'])
    merged = merged[merged['sales_2016'] > 0]
    
    merged['survived_2019'] = (merged['sales_2019'] > 0).astype(int)
    
    # Standardize
    merged['ec_std'] = (merged['ec'] - merged['ec'].mean()) / merged['ec'].std()
    merged['clustering_std'] = (merged['clustering'] - merged['clustering'].mean()) / merged['clustering'].std()
    merged['civic_std'] = (merged['civic'] - merged['civic'].mean()) / merged['civic'].std()
    merged['log_employees'] = np.log1p(merged['emp_2016'])
    merged['state'] = merged['fips'].astype(str).str[:2]
    merged['naics2'] = merged['naics'].astype(str).str[:2]
    
    formula = "survived_2019 ~ ec_std + clustering_std + civic_std + log_employees + C(state) + C(naics2)"
    model = smf.ols(formula=formula, data=merged).fit(cov_type='cluster', cov_kwds={'groups': merged['fips']})
    
    print(model.summary())
    
    return model


if __name__ == "__main__":
    summary_stats()