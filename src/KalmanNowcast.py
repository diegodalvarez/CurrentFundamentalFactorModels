# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:31:46 2026

@author: Diego
"""

import os
import pickle
import numpy as np
import pandas as pd
import datetime as dt
from   pykalman import KalmanFilter

from tqdm import tqdm

class KalmanNowcasting:
    
    def __init__(self) -> None: 
        
        self.src_path = os.getcwd()
        self.repo_path  = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path  = os.path.join(self.repo_path, "data")
        self.db_path    = os.path.join(self.data_path, "DataBase")
        self.quart_path = os.path.join(self.db_path, "Quarterlyized") 
        self.model_path = os.path.join(self.data_path, "Models")
        
        self.sample_tickers = ['SPX Index', 'RTY Index', 'NDX Index']
        
    def _get_fund_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df_out = (df
                .sort_values("quarter")
                .assign(diff_val = lambda x: x.value.diff()))
        
        return df_out
    
    def _get_generic_kalman_filter(self, y: np.ma, start_date: dt.date) -> KalmanFilter:
        
        y_masked = np.ma.masked_invalid(y)
        
        kf = KalmanFilter(
            transition_matrices=np.array([[1]]),
            observation_matrices=np.array([[1]]),
            
            # how much EBIT can move daily
            transition_covariance=np.array([[1.0]]),
            
            # how noisy reported EBIT is
            observation_covariance=np.array([[10.0]]),
            
            initial_state_mean=y[~np.isnan(y)][0],
            initial_state_covariance=np.array([[1.0]]))
        
        state_means, state_cov = kf.smooth(y_masked)
        
        out_dict = {
            "model"      : kf,
            "state_means": state_means,
            "state_cov"  : state_cov,
            "start_date" : start_date}
        
        return out_dict
                
    def get_generic_kalman(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Generic Kalman Models")
        
        fund_path = os.path.join(
            self.quart_path, 
            "QuarterlyFundamentalEquityIndexData.parquet")
        
        rtn_path = os.path.join(
            self.model_path,
            "BaselineModels",
            "VolTargetedReturns.parquet")

        ticker_path = os.path.join(
            self.data_path, 
            "EquityForecastTickerGuide.xlsx")
        
        out_path = os.path.join(
            self.model_path, "BaselineModels", "KalmanModels.pkl")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_ticker = (pd
            .read_excel(io = ticker_path, sheet_name = "FactorGuide")
            .loc[lambda x: x.est_field != "BEST_PE_RATIO"])
        
        real_fields = df_ticker.real_field.drop_duplicates().sort_values().to_list()
        
        df_raw_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.ticker.isin(self.sample_tickers)]
                .assign(quarter = lambda x: x.date.dt.year.astype(str) + "-Q" + x.date.dt.quarter.astype(str)))
        
        df_raw_fundamental = (pd
                .read_parquet(path = fund_path)
                .loc[lambda x: x.ticker.isin(self.sample_tickers)]
                .loc[lambda x: x.stat_method == "last"]
                .loc[lambda x: x.field.isin(real_fields)]
                .assign(group_var = lambda x: x.ticker + "-" + x.field))
        
        group_vars  = df_raw_fundamental.group_var.drop_duplicates().sort_values().to_list()
        vol_targets = df_raw_rtn.vol_target.drop_duplicates().sort_values().to_list()
        models      = {}
        
        for group_var in tqdm(group_vars):
            for vol_target in vol_targets:
                
                ticker, field = group_var.split("-")
                
                df_rtn = (df_raw_rtn
                        .loc[lambda x: x.vol_target == vol_target]
                        .loc[lambda x: x.ticker == ticker]
                        .assign(quarter = lambda x: x.date.dt.year.astype(str) + "-Q" + x.date.dt.quarter.astype(str))
                        .sort_values("date")
                        .assign(lag_rtn = lambda x: x.rtn_val.shift())
                        .dropna())
                
                df_map_guide = (df_rtn
                    [["date", "quarter", "ticker"]]
                    .groupby(["ticker", "quarter"])
                    .tail(1))
                
                df_fundamental = (df_raw_fundamental
                        .loc[lambda x: x.group_var == group_var]
                        .drop(columns = ["group_var"])
                        .merge(right = df_map_guide, how = "inner", on = ["quarter", "ticker"])
                        .sort_values("date")
                        .assign(val_diff = lambda x: x.value.diff())
                        .dropna())
                
                start_date = max([df_rtn.date.min(), df_fundamental.date.min()])
                
                df_combined = (df_fundamental
                    .merge(right = df_rtn, how = "right", on = ["date", "ticker", "quarter"])
                    .loc[lambda x: x.date >= start_date]
                    .sort_values("date"))
                
                y                  = np.ma.masked_invalid(df_combined.val_diff)
                model_name         = f"{ticker}-{field}-{vol_target}-kalman"
                models[model_name] = self._get_generic_kalman_filter(y, start_date)
                
        with open(out_path, "wb") as f: pickle.dump(models, f)
        
def main() -> None: 
        
    kf_nowcasting = KalmanNowcasting()
    kf_nowcasting.get_generic_kalman()
    
if __name__ == "__main__": main()