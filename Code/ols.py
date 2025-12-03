# run OLS regression as a baseline
# Daman Dhaliwal

# import libraries
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

from data_prep import merged_survival, merged_combined

def run_ols_survival(formula, overwrite = False):
    data = merged_survival(overwrite = overwrite)

    data = data.to_pandas()

    required_cols = ['survived_2024', 'ec', 'employees', 'sales', 'fips', 'naics', 'naics2']
    initial_len = len(data)
    data = data.dropna(subset=required_cols)
    print(f"Dropped {initial_len - len(data)} rows due to missing required columns.")

    data = data.drop('firm_age', axis = 1)

    # standardize all continuous variables - employees, sales, ec, clustering, civic
    data['ec_std'] = (data['ec'] - data['ec'].mean()) / data['ec'].std()
    data['clustering_std'] = (data['clustering'] - data['clustering'].mean()) / data['clustering'].std()
    data['civic_std'] = (data['civic'] - data['civic'].mean()) / data['civic'].std()
    data['log_employees'] = np.log1p(data['employees'])
    data['log_sales'] = np.log1p(data['sales'])
    data['state'] = data['fips'].str[:2]

    model = smf.ols(formula = formula, data = data).fit(cov_type = 'cluster', cov_kwds = {'groups': data['fips']})
    print(model.summary())

    return model

def run_ols_main(formula, overwrite = False):
    df = merged_combined(overwrite = overwrite)
    df = df.to_pandas()

    # prep baseline 2019
    cols = ['abi', 'fips', 'sales', 'employees', 'naics', 'ec', 'clustering', 'civic']
    cols = [c for c in cols if c in df.columns]
    
    data_2019 = df[df['file_year'] == 2019][cols].copy()
    data_2019 = data_2019.rename(columns={'sales': 'sales_2019', 'employees': 'emp_2019'})

    # prep outcome 2024
    data_2024 = df[df['file_year'] == 2024][['abi', 'sales']].copy()
    data_2024 = data_2024.rename(columns={'sales': 'sales_2024'})

    # merge and fill
    merged = data_2019.merge(data_2024, on='abi', how='left')
    merged['sales_2024'] = merged['sales_2024'].fillna(0)

    # filter for survivors only (intensive margin)
    merged = merged[merged['sales_2024'] > 0]

    # drop missing
    merged = merged.dropna(subset = ['sales_2019', 'ec', 'fips', 'emp_2019', 'naics'])
    merged = merged[merged['sales_2019'] > 0]

    # transformations
    merged['log_sales_2019'] = np.log1p(merged['sales_2019'])
    merged['log_sales_2024'] = np.log1p(merged['sales_2024'])
    merged['log_sales_change'] = merged['log_sales_2024'] - merged['log_sales_2019']

    # standardize and log controls
    merged['ec_std'] = (merged['ec'] - merged['ec'].mean()) / merged['ec'].std()
    merged['clustering_std'] = (merged['clustering'] - merged['clustering'].mean()) / merged['clustering'].std()
    merged['civic_std'] = (merged['civic'] - merged['civic'].mean()) / merged['civic'].std()
    
    merged['log_emp_2019'] = np.log1p(merged['emp_2019'])
    merged['state'] = merged['fips'].astype(str).str[:2]
    merged['naics2'] = merged['naics'].astype(str).str[:2]

    model = smf.ols(formula = formula, data = merged).fit(cov_type = 'cluster', cov_kwds = {'groups': merged['fips']})
    print(model.summary())

    return model

# Survival formulas
SURV_EC = "survived_2024 ~ ec_std + log_employees + C(state) + C(naics2)"
SURV_COH = "survived_2024 ~ clustering_std + log_employees + C(state) + C(naics2)"
SURV_CIV = "survived_2024 ~ civic_std + log_employees + C(state) + C(naics2)"
SURV_JOINT = "survived_2024 ~ ec_std + clustering_std + civic_std + log_employees + C(state) + C(naics2)"

# Growth formulas
GROWTH_EC = "log_sales_change ~ ec_std + log_sales_2019 + C(state) + C(naics2)"
GROWTH_COH = "log_sales_change ~ clustering_std + log_sales_2019 + C(state) + C(naics2)"
GROWTH_CIV = "log_sales_change ~ civic_std + log_sales_2019 + C(state) + C(naics2)"
GROWTH_JOINT = "log_sales_change ~ ec_std + clustering_std + civic_std + log_sales_2019 + C(state) + C(naics2)"

if __name__ == "__main__":
    run_ols_survival(SURV_JOINT)
    run_ols_survival(SURV_EC)
    run_ols_survival(SURV_COH)
    run_ols_survival(SURV_CIV)
    run_ols_main(GROWTH_JOINT)
    run_ols_main(GROWTH_EC)
    run_ols_main(GROWTH_COH)
    run_ols_main(GROWTH_CIV)