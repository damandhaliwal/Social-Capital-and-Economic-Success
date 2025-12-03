# Data load and cleaning
# Daman Dhaliwal

# import libraries
import polars as pl
import pandas as pd
import numpy as np
from utils import paths
import os
import glob
import re


def load_data(overwrite = False):
    path = paths()
    output_dir = path['data']

    combined_path = os.path.join(output_dir, "business_panel_full.parquet")
    survival_path = os.path.join(output_dir, "business_survival_2019.parquet")

    if not overwrite:
        combined = pl.read_parquet(combined_path)
        survival = pl.read_parquet(survival_path)
        return combined, survival

    # Otherwise, rebuild from source
    data_dir = paths()['data_input']
    bus_data_dir = os.path.join(data_dir, 'business_data')
    search_pattern = os.path.join(bus_data_dir, "*_Business_Academic*.txt")
    filenames = glob.glob(search_pattern)

    dfs = []

    for file in filenames:
        print(file)
        year = int(re.search(r'(\d{4})_Business', file).group(1))
        df = pl.read_csv(
            file,
            encoding='utf8-lossy',
            ignore_errors=True,
        )
        # Normalize column names to uppercase
        df = df.rename({col: col.upper() for col in df.columns})

        df = df.select([
            pl.col("ABI").cast(pl.String).alias("abi"),
            pl.col("FIPS CODE").cast(pl.String).str.zfill(5).alias("fips"),
            pl.col("BUSINESS STATUS CODE").cast(pl.String).alias("status_code"),
            pl.col("YEAR ESTABLISHED").cast(pl.Int32, strict=False).alias("year_established"),
            pl.col("EMPLOYEE SIZE (5) - LOCATION").cast(pl.Int32, strict=False).alias("employees"),
            pl.col("SALES VOLUME (9) - LOCATION").cast(pl.Float64, strict=False).alias("sales"),
            pl.col("PRIMARY NAICS CODE").cast(pl.String).alias("naics"),
            pl.lit(year).alias("file_year"),
        ]).filter(
            pl.col("abi").is_not_null() &
            pl.col("fips").is_not_null()
        )
        dfs.append(df)

    # Combine into full panel dataset
    combined = pl.concat(dfs)
    print(combined.head())

    # building a cleaned survival dataset for survival analysis
    baseline = combined.filter(pl.col("file_year") == 2019).select([
        "abi", "fips", "employees", "sales", "naics", "year_established"
    ])

    # Get ABIs present in each year
    abis_by_year = {}
    for year in [2020, 2021, 2022, 2023, 2024]:
        abis_by_year[year] = set(
            combined.filter(pl.col("file_year") == year)["abi"].to_list()
        )

    # Add survival flags
    survival = baseline.with_columns([
        pl.col("abi").is_in(abis_by_year[2020]).cast(pl.Int8).alias("survived_2020"),
        pl.col("abi").is_in(abis_by_year[2021]).cast(pl.Int8).alias("survived_2021"),
        pl.col("abi").is_in(abis_by_year[2022]).cast(pl.Int8).alias("survived_2022"),
        pl.col("abi").is_in(abis_by_year[2023]).cast(pl.Int8).alias("survived_2023"),
        pl.col("abi").is_in(abis_by_year[2024]).cast(pl.Int8).alias("survived_2024"),
        pl.col("naics").str.slice(0, 2).alias("naics2"),
        (2019 - pl.col("year_established")).alias("firm_age"),
    ])

    # save
    os.makedirs(output_dir, exist_ok=True)

    # save parquet files for faster loading later
    combined.write_parquet(combined_path)
    survival.write_parquet(survival_path)

    # save full csv
    combined.write_csv(os.path.join(output_dir, "business_panel_full.csv"))

    return combined, survival

# social capital data
def load_social_capital():
    path = paths()
    sc_path = os.path.join(path['data_input'], 'OI_data', 'social_capital_county.csv')
    
    sc = pl.read_csv(sc_path)
    
    # Select and rename columns
    sc = sc.select([
        pl.col("county").cast(pl.String).str.zfill(5).alias("fips"),
        pl.col("ec_county").alias("ec"),
        pl.col("clustering_county").alias("clustering"),
        pl.col("civic_organizations_county").alias("civic"),
    ])
    
    return sc

# merge the datasets
def merged_survival(overwrite = False):
    path = paths()
    output_path = os.path.join(path['data'], "survival_merged.parquet")
    
    if not overwrite and os.path.exists(output_path):
        return pl.read_parquet(output_path)
    
    _ , survival = load_data(overwrite=False)
    sc = load_social_capital()
    
    # Merge datasets on FIPS code
    merged = survival.join(sc, on="fips", how="left")
    
    # Check merge rate
    n_matched = merged.filter(pl.col("ec").is_not_null()).height
    print(f"Merge rate: {n_matched / len(merged):.1%}")
    
    # Save
    merged.write_parquet(output_path)
    
    return merged

def merged_combined(overwrite = False):
    path = paths()
    output_path = os.path.join(path['data'], "combined_merged.parquet")
    
    if not overwrite and os.path.exists(output_path):
        return pl.read_parquet(output_path)
    
    combined, _ = load_data(overwrite=False)
    sc = load_social_capital()
    
    # Merge datasets on FIPS code
    merged = combined.join(sc, on="fips", how="left")
    
    # Check merge rate
    n_matched = merged.filter(pl.col("ec").is_not_null()).height
    print(f"Merge rate: {n_matched / len(merged):.1%}")
    
    # Save
    merged.write_parquet(output_path)
    
    return merged