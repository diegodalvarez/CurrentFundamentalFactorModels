# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:50:22 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

class DataPrep:
    
    def __init__(self) -> None: 
        
        self.src_path  = os.getcwd()
        self.repo_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.repo_path, "data")
        self.db_path   = os.path.join(self.data_path, "DataBase")
        self.prep_path = os.path.join(self.db_path  , "PrepData")
        
        if not os.path.exists(self.prep_path): os.makedirs(self.prep_path)
        
        self.sample_tickers = ['SPX Index', 'RTY Index', 'NDX Index']
        self.est_vol_window = 10
        self.rtn_vol_target = 0.1
        self.rtn_vol_window = 100
        
    def _get_vol_rtn(self, df: pd.DataFrame, target: float = 0.1, window: int = 100) -> pd.DataFrame: 
    
        df_out = (df
            .sort_values("date")
            .assign(
                vol         = lambda x: x.value.ewm(span = window, adjust = False).std(),
                vol_rtn     = lambda x: x.value * (target / (x.vol * np.sqrt(252))),
                lag_vol_rtn = lambda x: x.value * (target / (x.vol.shift() * np.sqrt(252))))
            .drop(columns = ["vol"])
            .set_index("date"))
        
        return df_out
        
    def vol_target_rtn(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Vol. Targeted Returns")
        
        daily_path  = os.path.join(self.db_path, "DailyData")
        rtn_path    = os.path.join(daily_path, "DailyReturnsEquityIndexData.parquet")
        out_path    = os.path.join(self.prep_path, "VolTargetedReturns.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.field == "DAY_TO_DAY_TOT_RETURN_GROSS_DVDS"]
                .drop(columns = ["field"])
                .assign(value = lambda x: x.value / 100)
                .groupby(["date", "ticker"])
                .head(1) ## investigate
                .groupby("ticker")
                .apply(self._get_vol_rtn, self.rtn_vol_target, self.rtn_vol_window)
                .reset_index()
                .dropna()
                .rename(columns = {"value": "rtn"})
                .melt(id_vars = ["date", "ticker"], var_name = "vol_target", value_name = "rtn_val"))
        
        if verbose: print("Saving data\n")
        df_rtn.to_parquet(path = out_path, engine = "pyarrow")

    def _get_est_adj(self, df: pd.DataFrame, vol_window: int) -> pd.DataFrame: 
        
        df_diff = (df
                .pivot(index = "date", columns = "field", values = "value")
                .diff())
        
        df_adj = (df_diff
                .apply(lambda x: x / x.ewm(span = vol_window, adjust = False).std())
                .reset_index()
                .melt(id_vars = "date")
                .dropna()
                .assign(field_type = "adj_val"))
        
        df_val = (df_diff
                .reset_index()
                .melt(id_vars = "date")
                .assign(field_type = "val")
                .dropna())
        
        df_out = (pd
                .concat([df_adj, df_val]))
        
        return df_out

    def prep_analyst_est_fundamental(self, verbose: bool = True) -> None:
        
        if verbose: print("Getting Estimated Changes Factor")

        daily_path  = os.path.join(self.db_path, "DailyData")
        est_path    = os.path.join(daily_path, "DailyEstimateEquityIndexData.parquet")
        ticker_path = os.path.join(self.data_path, "EquityForecastTickerGuide.xlsx")
        out_path    = os.path.join(self.prep_path, "AdjAnalystEstimate.parquet")

        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None

        df_ticker_map = (pd
            .read_excel(io = ticker_path, sheet_name = "FactorGuide")
            .loc[lambda x: x.est_field != "BEST_PE_RATIO"])

        est_fields = df_ticker_map.est_field.drop_duplicates().sort_values().to_list()

        df_adj_est = (pd
                .read_parquet(path = est_path, engine = "pyarrow")
                .loc[lambda x: x.ticker.isin(self.sample_tickers)]
                .loc[lambda x: x.field.isin(est_fields)]
                .groupby(["date", "ticker", "field"])
                .head(1)
                .reset_index()
                .groupby("ticker")
                .apply(self._get_est_adj, self.est_vol_window)
                .reset_index()
                .drop(columns = ["level_1"]))

        if verbose: print("Saving analyst estimate\n")
        df_adj_est.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_date_mapper(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Date Mapper")
        
        out_path = os.path.join(self.prep_path, "DateMapper.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        path = os.path.join(self.prep_path, "VolTargetedReturns.parquet")
        df   = (pd
                .read_parquet(path = path, engine = "pyarrow")
                [["date"]]
                .drop_duplicates())
        
        df["quarter"] = (
            df["date"].dt.year.astype(str)
            + "-Q"
            + df["date"].dt.quarter.astype(str))
        
        if verbose: print("Saving data\n")
        df.to_parquet(path = out_path, engine = "pyarrow")

def main() -> None: 

    data_prep = DataPrep()
    #data_prep.vol_target_rtn()
    #data_prep.prep_analyst_est_fundamental()
    #data_prep.get_date_mapper()

if __name__ == "__main__": main()