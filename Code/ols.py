# run OLS regression as a baseline
# Daman Dhaliwal

# import libraries
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

from data_prep import merged_survival, merged_combined

def run_ols_survival(overwrite = False):
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

    formula = "survived_2024 ~ ec_std + log_sales + log_employees + C(state) + C(naics2)"

    model = smf.ols(formula = formula, data = data).fit(cov_type = 'cluster', cov_kwds = {'groups': data['fips']})

    print(f"Social Capital (ec_std) Coefficient: {model.params['ec_std']:.5f}")
    print(f"Standard Error:                      {model.bse['ec_std']:.5f}")
    print(f"P-value:                             {model.pvalues['ec_std']:.5f}")
    print(f"Confidence Interval (95%):           [{model.conf_int().loc['ec_std'][0]:.5f}, {model.conf_int().loc['ec_std'][1]:.5f}]")
    
    print("-" * 60)
    print(f"R-squared:                           {model.rsquared:.4f}")
    print(f"Observations:                        {model.nobs:,.0f}")

    print(model.summary())

    return model

if __name__ == "__main__":
    run_ols_survival()
    