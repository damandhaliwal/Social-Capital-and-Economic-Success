# DoWhy Robustness Checks for Causal Inference
# Daman Dhaliwal

# import libraries
import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel

# Import your data loader
from data_prep import merged_survival

def run_dowhy_robustness(overwrite = False):
    df = merged_survival(overwrite = overwrite)
    
    pdf = df.to_pandas()

    subset_cols = ['survived_2022', 'ec', 'employees', 'naics2', 'fips', 'firm_age']
    pdf = pdf.dropna(subset=subset_cols)

    # Standardize & Prep
    pdf['ec_std'] = (pdf['ec'] - pdf['ec'].mean()) / pdf['ec'].std()
    pdf['state'] = pdf['fips'].str[:2]
    
    model = CausalModel(
        data=pdf,
        treatment='ec_std',
        outcome='survived_2022',
        common_causes=['employees', 'firm_age'], 
        effect_modifiers=['naics2', 'state'] 
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True
    )
    print(f"   Baseline Estimate: {estimate.value:.5f}")

    res_placebo = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter"
    )
    print(res_placebo)

if __name__ == "__main__":
    run_dowhy_robustness()