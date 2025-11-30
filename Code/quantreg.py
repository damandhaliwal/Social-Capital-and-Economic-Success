# Quantile Regression with entire dataset
# Daman Dhaliwal

# import libraries
import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from tqdm import tqdm

from data_prep import merged_combined

def run_quantreg(overwrite = False):
    df = merged_combined(overwrite = overwrite)
    df = df.to_pandas()

    data_2019 = df[df['file_year'] == 2019][['abi', 'fips', 'sales', 'employees', 'naics', 'ec']].copy()
    data_2019 = data_2019.rename(columns={'sales': 'sales_2019', 'employees': 'emp_2019'})

    data_2024 = df[df['file_year'] == 2024][['abi', 'sales']].copy()
    data_2024 = data_2024.rename(columns={'sales': 'sales_2024'})

    merged = data_2019.merge(data_2024, on='abi', how='left')
    merged['sales_2024'] = merged['sales_2024'].fillna(0)

    # compute sales change
    merged['survived'] = (merged['sales_2024'] > 0).astype(int)

    merged = merged.dropna(subset = ['sales_2019', 'ec', 'fips', 'emp_2019', 'naics'])
    merged = merged[merged['sales_2019'] > 0]

    # log sales
    merged['log_sales_2019'] = np.log1p(merged['sales_2019'])
    merged['log_sales_2024'] = np.log1p(merged['sales_2024'])
    merged['log_sales_change'] = merged['log_sales_2024'] - merged['log_sales_2019']

    # standardize independent variables
    merged['ec_std'] = (merged['ec'] - merged['ec'].mean()) / merged['ec'].std()
    merged['log_emp_2019'] = np.log1p(merged['emp_2019'])
    merged['naics2'] = merged['naics'].str[:2]

    # define quantiles from 0.05 to 0.95
    quantiles = np.arange(0.05, 1.00, 0.05)

    results = []

    for q in tqdm(quantiles, desc="   Quantiles", ncols=70):
        X = merged[['ec_std', 'log_emp_2019', 'log_sales_2019']].assign(const=1)
        model = QuantReg(merged['log_sales_change'], X)
        res = model.fit(q=q)

        coef = res.params['ec_std']
        se = res.bse['ec_std']
        pval = res.pvalues['ec_std']

        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"   {q:<10.2f} {coef:>12.4f} {se:>12.4f} {pval:>10.4f} {sig:>5}")

        results.append({'quantile': q, 'coef': coef, 'se': se, 'pval': pval})

    results_df = pd.DataFrame(results)

    coef_10 = results_df[results_df['quantile'] == 0.10]['coef'].values[0]
    coef_90 = results_df[results_df['quantile'] == 0.90]['coef'].values[0]

    print(results_df)

    return results_df, merged

if __name__ == "__main__":
    run_quantreg()