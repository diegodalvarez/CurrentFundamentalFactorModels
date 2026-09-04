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
        self.prep_path  = os.path.join(self.db_path, "PrepData")
        
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
        
        '''
        Runs a regression to predict fundamental data using lagged values
        '''
    
        if verbose: print("Getting Single Factor OLS Models")    
    
        quarter_path = os.path.join(self.db_path, "Quarterlyized")
        out_path     = os.path.join(self.base_path, "SingleVarPredictiveModels.pkl")

        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
    
        quart_path = os.path.join(
            quarter_path,
            "QuarterlyFundamentalEquityIndexData.parquet")
        
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
            
            ols_model = (sm
                    .OLS(
                        endog = df_input[field_name],
                        exog  = sm.add_constant(df_input[["lag_" + field_name]]))
                    .fit())
            
            models[name] = ols_model
        
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f:
            pickle.dump(models, f)
            
    def get_multi_factor_regression(self, verbose: bool = True) -> None: 
        
        '''
        Runs a regression using all available data (lagged) to predict 
        fundamental forecast
        '''
        
        if verbose: print("Getting Multi-Factor Prediction")
        
        quarter_path = os.path.join(self.db_path, "Quarterlyized")
        out_path     = os.path.join(self.base_path, "MultiVarPredictiveModels.pkl")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        quart_path = os.path.join(
            quarter_path,
            "QuarterlyFundamentalEquityIndexData.parquet")
        
        df_raw_values = (pd
                .read_parquet(path = quart_path, engine = "pyarrow")
                .loc[lambda x: x.ticker.isin(self.sample_tickers)]     # security constraint
                .loc[lambda x: x.stat_method.isin(self.stat_method)]   # stat method constraint
                .assign(ticker = lambda x: x.ticker.str.split(" ").str[0])
                .drop(columns = ["stat_method"])
                .set_index("quarter")
                .groupby(["ticker", "field"])
                .apply(lambda x: x.sort_index().value.diff())
                .to_frame(name = "value")
                .reset_index()
                .dropna())
        
        df_lagged_value = (df_raw_values
                .set_index("quarter")
                .groupby(["ticker", "field"])
                .apply(lambda x: x.sort_index().value.shift())
                .to_frame(name = "lagged_value")
                .reset_index())
        
        df_guide = (df_raw_values
                [["ticker", "field"]]
                .drop_duplicates())
        
        models = {}
        
        for i, row in df_guide.iterrows():
            
            row_dict   = row.to_dict()
            tmp_ticker = row_dict["ticker"]
            tmp_field  = row_dict["field"]
            
            df_endog = (df_raw_values
                    .loc[lambda x: (x.ticker == tmp_ticker) & (x.field == tmp_field)]
                    .drop(columns = ["ticker", "field"])
                    .rename(columns = {"value": tmp_field}))
            
            df_exog = (df_lagged_value
                    .loc[lambda x: (x.ticker == tmp_ticker) & (x.field != tmp_field)]
                    .assign(name = lambda x: "lag_" + x.field)
                    .pivot(index = "quarter", columns = "name", values = "lagged_value")
                    .dropna())
            
            df_combined = (df_endog
                           .merge(right = df_exog, how = "inner", on = ["quarter"]))
            
            model = (sm
                     .OLS(
                         endog = df_combined[tmp_field],
                         exog  = sm.add_constant(df_combined[df_exog.columns.to_list()]))
                     .fit())
            
            models[tmp_ticker + "-" + tmp_field] = model
        
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f:
            pickle.dump(models, f)
        
    def get_analyst_estimated_factor_exposures(self, verbose: bool = True) -> None:
        
        if verbose: print("Getting Estimated Changes Factor")
        
        ticker_path = os.path.join(self.data_path, "EquityForecastTickerGuide.xlsx")
        out_path    = os.path.join(self.base_path, "EstimatedFactorExposure.pkl")
        rtn_path    = os.path.join(self.prep_path, "VolTargetedReturns.parquet")
        adj_path    = os.path.join(self.prep_path, "AdjAnalystEstimate.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.ticker.isin(self.sample_tickers)]
                .loc[lambda x: x.vol_target != "lag_vol_rtn"])
        
        df_ticker_map = (pd
            .read_excel(io = ticker_path, sheet_name = "FactorGuide")
            .loc[lambda x: x.est_field != "BEST_PE_RATIO"])
        
        est_fields = df_ticker_map.est_field.drop_duplicates().sort_values().to_list()
        
        df_est_adj = (pd
              .read_parquet(path = adj_path, engine = "pyarrow")
              .loc[lambda x: x.field_type == "adj_val"]
              .drop(columns = ["field_type"])
              .loc[lambda x: x.field.isin(est_fields)])
        
        df_combined = (df_rtn
                .merge(right = df_est_adj, how = "inner", on = ["ticker", "date"])
                .assign(group_var = lambda x: x.ticker + "-" + x.vol_target + "-" + x.field))
        
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
    
    def get_single_factor_rtn_regression(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Single Factor Regression with Lagged Quarterly Returns")
        
        rtn_path     = os.path.join(self.prep_path, "VolTargetedReturns.parquet")
        date_path    = os.path.join(self.prep_path, "DateMapper.parquet")
        quarter_path = os.path.join(self.db_path, "Quarterlyized")
        out_path     = os.path.join(self.base_path, "SingleVarWithRtnPredictiveModels.pkl")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        quart_path = os.path.join(
            quarter_path,
            "QuarterlyFundamentalEquityIndexData.parquet")
        
        df_fundamental = (pd
                .read_parquet(path = quart_path, engine = "pyarrow")
                .loc[lambda x: x.ticker.isin(self.sample_tickers)]     # security constraint
                .loc[lambda x: x.stat_method.isin(self.stat_method)]   # stat method constraint
                .assign(ticker = lambda x: x.ticker.str.split(" ").str[0]))
        
        df_date_mapper = (pd
                .read_parquet(path = date_path, engine = "pyarrow"))
        
        df_quarterly_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .loc[lambda x: x.ticker.isin(self.sample_tickers)]
                .loc[lambda x: x.vol_target != "lag_vol_rtn"]
                .assign(ticker = lambda x: x.ticker.str.split(" ").str[0])
                .merge(right = df_date_mapper, how = "inner", on = ["date"])
                .set_index("date")
                .groupby(["ticker", "vol_target", "quarter"])
                .agg(lambda x: (1 + x.sort_index()).prod() - 1)
                .reset_index()
                .set_index("quarter")
                .groupby(["ticker", "vol_target"])
                .apply(lambda x: x.rtn_val.sort_index().shift())
                .reset_index()
                .melt(id_vars = ["ticker", "vol_target"])
                .dropna()
                .rename(columns = {"value": "lag_rtn"}))

        df_fundamental_diff = (df_fundamental
                .drop(columns = ["stat_method"])
                .set_index("quarter")
                .groupby(["ticker", "field"])
                .apply(lambda x: x.sort_index().value.diff())
                .reset_index()
                .dropna())
        
        df_lagged_fundamental_diff = (df_fundamental_diff
                .set_index("quarter")
                .groupby(["ticker", "field"])
                .apply(lambda x: x.sort_index().value.shift())
                .to_frame(name = "lag_fundamental")
                .reset_index())
        
        df_combined = (df_fundamental_diff
                .merge(
                    right = df_lagged_fundamental_diff, 
                    how   = "inner", 
                    on    = ["ticker", "field", "quarter"])
                .rename(columns = {"value": "fundamental"})
                .merge(right = df_quarterly_rtn, how = "inner", on = ["ticker", "quarter"])
                .assign(group_var = lambda x: x.ticker + "-" + x.field + "-" + x.vol_target))
        
        group_vars = df_combined.group_var.drop_duplicates().sort_values().to_list()
        models     = {}
        
        for group_var in group_vars: 
            
            df_tmp = (df_combined
                    .loc[lambda x: x.group_var == group_var]
                    .drop(columns = ["group_var"])
                    .set_index("quarter")
                    .dropna())
            
            model = (sm
                    .OLS(
                        endog = df_tmp.fundamental,
                        exog  = sm.add_constant(df_tmp[["lag_fundamental", "lag_rtn"]]))
                    .fit())
            
            models[group_var] = model
            
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f: pickle.dump(models, f)
    
def main() -> None: 
        
    baseline_model = BaselineModels()
    #baseline_model.get_single_factor_regression()
    #baseline_model.get_multi_factor_regression()
    #baseline_model.get_analyst_estimated_factor_exposures()
    #baseline_model.get_single_factor_rtn_regression()
    
if __name__ == "__main__": main()