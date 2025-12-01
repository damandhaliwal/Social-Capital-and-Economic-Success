# DML Heterogeneity Analysis by Sub-Industries
# Daman Dhaliwal

# import libraries
import pandas as pd
import numpy as np
import doubleml as dml
from xgboost import XGBRegressor, XGBClassifier
import os

from data_prep import merged_survival 
from utils import paths

def run_sub_industry_dml(overwrite = False):
    data = merged_survival(overwrite = overwrite)

    data = data.to_pandas()

    required_cols = ['survived_2024', 'ec', 'employees', 'sales', 'fips']
    initial_len = len(data)
    data = data.dropna(subset=required_cols)
    print(f"Dropped {initial_len - len(data)} rows due to missing required columns.")

    data = data.drop('firm_age', axis = 1)

    # create naics 4 code for industry
    data['naics4'] = data['naics'].astype(str).str[:4]

    # standardize all continuous variables - employees, sales, ec, clustering, civic
    data['ec_std'] = (data['ec'] - data['ec'].mean()) / data['ec'].std()
    data['clustering_std'] = (data['clustering'] - data['clustering'].mean()) / data['clustering'].std()
    data['civic_std'] = (data['civic'] - data['civic'].mean()) / data['civic'].std()
    data['log_employees'] = np.log1p(data['employees'])
    data['log_sales'] = np.log1p(data['sales'])
    data['state'] = data['fips'].str[:2]

    # create dummies for controls (state only)
    data = pd.get_dummies(data, columns = ['state'], drop_first = True)

    sub_industries = data['naics4'].unique()

    results = []

    for ind in sub_industries:
        df = data[data['naics4'] == ind].copy()

        if len(df) < 500:
            continue

        print(ind, len(df))

        valid_cols = ['log_sales', 'ec_std', 'civic_std'] + [c for c in df.columns if c.startswith('state_')]
        valid_cols = [c for c in valid_cols if df[c].nunique() > 1]

        # setup DoubleML
        dml_data = dml.DoubleMLData(
            df,
            y_col = 'survived_2024',
            d_cols = 'clustering_std',
            x_cols = valid_cols
        )

        # define learners
        ml_l = XGBClassifier(n_estimators = 200, max_depth = 3, learning_rate = 0.1, n_jobs = -1, random_state = 42, use_label_encoder = False, eval_metric = 'logloss')
        ml_m = XGBRegressor(n_estimators = 200, max_depth = 3, learning_rate = 0.1, n_jobs = -1, random_state = 42)

        dml_model = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds = 5)
        dml_model.fit()

        coef = dml_model.coef[0]
        pval = dml_model.pval[0]

        results.append({
            'naics2': ind,
            'n': len(df),
            'coef': coef,
            'pval': pval,
            'signficant': '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else 'n.s.'
        })

    results_df = pd.DataFrame(results).sort_values('coef', ascending=False)
    print(results_df)
    path = paths()['data']
    filename = os.path.join(path, 'dml_sub_industry_results.csv')
    results_df.to_csv(filename, index=False)
    return results_df


if __name__ == "__main__":
    run_sub_industry_dml()