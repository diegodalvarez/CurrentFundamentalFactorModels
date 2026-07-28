# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 20:03:09 2026

@author: Diego
"""

import os 
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm

from tqdm import tqdm
tqdm.pandas()

class BaselineModels:
    
    def __init__(self) -> None: 
        
        self.src_path   = os.getcwd()
        self.repo_path  = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path  = os.path.join(self.repo_path, "data")
        self.db_path    = os.path.join(self.data_path, "DataBase")
        self.model_path = os.path.join(self.data_path, "Models")
        self.base_path  = os.path.join(self.model_path, "BaselineModels")
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            
        self.sample_tickers = ['SPX Index', 'RTY Index', 'NDX Index']
        self.stat_method    = ["last"]
        self.est_vol_window = 10
        self.rtn_vol_target = 0.1
        self.rtn_vol_window = 100
        
    def _get_diff(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df
                .sort_values("value")
                .assign(diff_val = lambda x: x.value.diff()))
        
        return df_out
    
    def get_single_factor_regression(self, verbose: bool = True) -> None:
    
        if verbose: print("Getting Single Factor OLS Models")    
    
        quarter_path = os.path.join(self.db_path, "Quarterlyized")
        out_path     = os.path.join(self.base_path, "PredictiveModels.pkl")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
    
        quart_path = os.path.join(
            quarter_path,
            "QuarterlyFundamentalEquityIndexData.parquet"
        )
        
        df_raw = (pd
                .read_parquet(path = quart_path, engine = "pyarrow")
                .loc[lambda x: x.ticker.isin(self.sample_tickers)]     # security constraint
                .loc[lambda x: x.stat_method.isin(self.stat_method)]   # stat method constraint
                .assign(ticker = lambda x: x.ticker.str.split(" ").str[0]))  
    
        df_diff = (df_raw
                .assign(name = lambda x: x.ticker + "-" + x.field + "-" + x.stat_method)
                .groupby("name")
                #.progress_apply(lambda group: self._get_diff(group))
                .apply(lambda group: self._get_diff(group))
                .reset_index()
                .dropna()
                .drop(columns = ["level_1"]))
        
        names  = df_diff.name.drop_duplicates().sort_values().to_list()
        models = {}
        
        for name in names: 
            
            field_name = name.split("-")[1]
            
            df_input = (df_diff
                    .loc[lambda x: x.name == name]
                    .set_index("quarter")
                    [["diff_val"]]
                    .assign(lag_val = lambda x: x.diff_val.shift())
                    .rename(columns = {
                        "diff_val": field_name,
                        "lag_val" : "lag_" + field_name})
                    .dropna())
            
            model = (sm
                    .OLS(
                        endog = df_input[field_name],
                        exog  = sm.add_constant(df_input.drop(columns = [field_name])))
                    .fit())
            
            models[name] = model
        
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f:
            pickle.dump(models, f)
    
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
        
    def get_estimated_factor_exposures(self, verbose: bool = True) -> None:
        
        if verbose: print("Getting Estimated Changes Factor")
        
        daily_path  = os.path.join(self.db_path, "DailyData")
        est_path    = os.path.join(daily_path, "DailyEstimateEquityIndexData.parquet")
        ticker_path = os.path.join(self.data_path, "EquityForecastTickerGuide.xlsx")
        out_path    = os.path.join(self.base_path, "EstimatedFactorExposure.pkl")
        rtn_path    =  os.path.join(self.base_path, "VolTargetedReturns.parquet")

        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.ticker.isin(self.sample_tickers)])

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
        
        df_combined = (df_rtn
                .merge(right = df_adj_est, how = "inner", on = ["ticker", "date"])
                .assign(group_var = lambda x: x.ticker + "-" + x.vol_target + "-" + x.field + "-" + x.field_type))
        
        groups = df_combined.group_var.drop_duplicates().sort_values().to_list()
        models = {}
        
        for group in tqdm(groups): 
            
            df_input = (df_combined
                    .loc[lambda x: x.group_var == group]
                    .set_index("date")
                    [["rtn_val", "value"]])
            
            model = (sm
                    .OLS(
                        endog = df_input.rtn_val,
                        exog  = sm.add_constant(df_input.value))
                    .fit())
            
            models[group] = model

        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f: pickle.dump(models, f)
        
    def get_vol_targeted_rtn(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Vol. Targeted Returns")
        
        daily_path  = os.path.join(self.db_path, "DailyData")
        rtn_path    = os.path.join(daily_path, "DailyReturnsEquityIndexData.parquet")
        out_path    = os.path.join(self.base_path, "VolTargetedReturns.parquet")
        
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
    
def main() -> None: 
        
    baseline_model = BaselineModels()
    #baseline_model.get_single_factor_regression()
    #baseline_model.get_vol_targeted_rtn()
    #baseline_model.get_estimated_factor_exposures()
    
    
if __name__ == "__main__": main()