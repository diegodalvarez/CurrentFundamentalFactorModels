# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 20:20:32 2026

@author: Diego
"""

import pandas as pd
import statsmodels.api as sm

class OLSModels:
    
    def _get_model_params(self, model: sm.OLS) -> pd.DataFrame: 
    
        df_param_val = model.params.to_frame(name = "param_val").reset_index()
        df_pvalues   = model.pvalues.to_frame(name = "pvalue").reset_index()
        df_tvalues   = model.tvalues.to_frame(name = "tvalue").reset_index()
    
        df_params = (df_param_val
            .merge(right = df_pvalues, how = "inner", on = ["index"])
            .merge(right = df_tvalues, how = "inner", on = ["index"])
            .assign(r2 = model.rsquared)
            .rename(columns = {"index": "param_name"}))
    
        return df_params
    
    def get_model_params(self, models: dict) -> pd.DataFrame: 
    
        params_list = []
        
        for model_key in models.keys():
        
            tmp_model = models[model_key]
            df_add    = self._get_model_params(tmp_model).assign(name = model_key)
            params_list.append(df_add)
        
        df_params = pd.concat(params_list)
        return df_params