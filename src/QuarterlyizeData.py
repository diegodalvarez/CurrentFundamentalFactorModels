# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:04:49 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

class Quarterlizer:
    
    def __init__(self) -> None: 
        
        self.src_path   = os.getcwd()
        self.repo_path  = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path  = os.path.join(self.repo_path, "data")
        self.db_path    = os.path.join(self.data_path, "DataBase")
        self.quart_path = os.path.join(self.db_path, "Quarterlyized")
        
        if os.path.exists(self.quart_path) == False: 
            os.makedirs(self.quart_path)
            
    def _get_quarterly(
        self,
        df: pd.DataFrame,
        method: str = "mode"
    ) -> pd.DataFrame:
        """
        Replace all values within each (group_var, quarter) using the
        specified aggregation method.
    
        Parameters
        ----------
        method : {"mode", "first", "last", "min", "max", "mean", "median"}
        """
    
        df = df.copy()
    
        df["quarter"] = (
            df["date"].dt.year.astype(str)
            + "-Q"
            + df["date"].dt.quarter.astype(str)
        )
    
        if method == "mode":
    
            df_replace = (
                df.groupby(["group_var", "quarter", "value"], as_index=False)
                  .size()
                  .sort_values(
                      ["group_var", "quarter", "size"],
                      ascending=[True, True, False]
                  )
                  .drop_duplicates(["group_var", "quarter"])
                  [["group_var", "quarter", "value"]]
                  .rename(columns={"value": "replace_val"})
            )
    
        elif method == "first":
    
            df_replace = (
                df.sort_values("date")
                  .drop_duplicates(["group_var", "quarter"])
                  [["group_var", "quarter", "value"]]
                  .rename(columns={"value": "replace_val"})
            )
    
        elif method == "last":
    
            df_replace = (
                df.sort_values("date")
                  .drop_duplicates(["group_var", "quarter"], keep="last")
                  [["group_var", "quarter", "value"]]
                  .rename(columns={"value": "replace_val"})
            )
    
        elif method in ["min", "max", "mean", "median"]:
    
            df_replace = (
                df.groupby(["group_var", "quarter"], as_index=False)
                  .agg(replace_val=("value", method))
            )
    
        else:
            raise ValueError(f"Unknown method: {method}")
    
        return df_replace

    def _quarterlize(
            self, 
            path_name: str, 
            freq     : str, 
            files    : list,
            verbose  : bool = True) -> None:
        
        daily_path = os.path.join(self.data_path, "DataBase", path_name)
        
        if verbose: print("Running Quarterly-izer ")
        
        for file in files: 
            
            if verbose: print("Working on {} {} data".format(freq, file))
            
            tmp_path = os.path.join(daily_path, "{}{}.parquet".format(freq, file))
            out_path = os.path.join(self.quart_path, "{}{}.parquet".format(freq, file))
            
            if os.path.exists(out_path) == True: 
                if verbose: print("Already have data\n")
                continue
                
            else:
            
                if verbose: print("Calculating data")
                
                df_raw = (pd
                          .read_parquet(path = tmp_path, engine = "pyarrow")
                          .assign(group_var = lambda x: x.ticker + "-" + x.field))
                
                methods  = ["mode", "first", "last", "min", "max", "mean", "median"]
                df_lists = []
                
                for method in methods:
                    
                    if verbose: print("Stat: {}".format(method))
                    
                    df_add = (self
                              ._get_quarterly(df = df_raw, method = method)
                              .assign(
                                  ticker      = lambda x: x.group_var.str.split("-").str[0],
                                  field       = lambda x: x.group_var.str.split("-").str[1],
                                  stat_method = method)
                              .drop(columns = ["group_var"])
                              .rename(columns = {"replace_val": "value"}))
                    
                    df_lists.append(df_add)
                
                df_out = pd.concat(df_lists)
                
                if verbose: print("Saving data\n")
                df_out.to_parquet(path = out_path, engine = "pyarrow")
            
    def quarterlize(self) -> None: 
        
        daily_files = [
            "EstimateEquityIndexData", "FundamentalEquityIndexData",
            "MarketEquityIndexData", "MarketETFData"]
        
        self._quarterlize("DailyData", "Daily", daily_files)
        self._quarterlize("QuarterlyData", "Quarterly", ["FundamentalEquityIndexData"])
        
    def _get_quarterly_rtn(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df
                .sort_values("date")
                .assign(cuml_rtn = lambda x: np.cumprod(1 + x.value) - 1)
                .loc[lambda x: x.date == x.date.min()])
        
        return df_out
        
    def get_quarterly_returns(self, verbose: bool = True) -> None:
        
        path     = os.path.join(self.data_path, "DataBase", "DailyData")
        idx_path = os.path.join(path, "DailyReturnsEquityIndexData.parquet")
        etf_path = os.path.join(path, "DailyReturnsETFData.parquet")

        idx_out_path = os.path.join(self.quart_path, "QuarterlyIndexReturns.parquet")
        if os.path.exists(idx_out_path):
            if verbose: print("Already have Index Data\n")
            return None
        
        else: 

            if verbose: print("Index Return Calculation")
            df_idx = (pd
                    .read_parquet(path = idx_path, engine = "pyarrow")
                    [["date", "ticker", "value"]]
                    .drop_duplicates()
                    .assign(
                        value   = lambda x: x.value / 100,
                        quarter = lambda x: 
                            x.date.dt.year.astype(str) + 
                            "-Q" + 
                            x.date.dt.quarter.astype(str),
                        group_var = lambda x: x.ticker + "+" + x.quarter)                
                    .groupby("group_var")
                    .progress_apply(lambda group: self._get_quarterly_rtn(group))
                    .reset_index(drop = True))
                
            if verbose: print("Saving data\n")
            df_idx.to_parquet(path = idx_out_path, engine = "pyarrow")
                
        etf_out_path = os.path.join(self.quart_path, "QuarterlyETFReturns.parquet")
        if os.path.exists(etf_out_path):
            if verbose: print("Already have ETF Data\n")
            return None
        
        else:
        
            if verbose: print("ETF Return Calculation")
            df_etf = (pd
                    .read_parquet(path = etf_path, engine = "pyarrow")
                    .drop(columns = ["field"])
                    .assign(
                        value   = lambda x: x.value / 100,
                        quarter = lambda x: 
                            x.date.dt.year.astype(str) + 
                            "-Q" + 
                            x.date.dt.quarter.astype(str),
                        group_var = lambda x: x.ticker + "+" + x.quarter)
                    .groupby("group_var")
                    .progress_apply(lambda group: self._get_quarterly_rtn(group))
                    .reset_index(drop = True))
            
            if verbose: print("Saving data\n")
            df_etf.to_parquet(path = etf_out_path, engine = "pyarrow")
        
def main() -> None: 
        
    quarterlizer = Quarterlizer()
    quarterlizer.quarterlize()
    quarterlizer.get_quarterly_returns()
    
if __name__ == "__main__": main()