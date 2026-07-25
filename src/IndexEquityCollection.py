import os
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt

from tqdm import tqdm
tqdm.pandas()

class DataCollection:
    
    def __init__(self, makedirs: bool = True) -> None:
        
        self.src_path   = os.getcwd()
        self.repo_path  = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path  = os.path.join(self.repo_path, "data")
        self.raw_path   = os.path.join(self.data_path, "RawData")
        self.daily_path = os.path.join(self.data_path, "DailyData")
        self.quart_path = os.path.join(self.data_path, "QuarterlyData")

        if os.path.exists(self.data_path) == False and makedirs == True: 
            os.makedirs(self.data_path)
            
        if os.path.exists(self.raw_path) == False and makedirs == True: 
            os.makedirs(self.raw_path)
            
        if os.path.exists(self.daily_path) == False and makedirs == True: 
            os.makedirs(self.daily_path)
            
        if os.path.exists(self.quart_path) == False and makedirs == True: 
            os.makedirs(self.quart_path)
            
        self.start_date = dt.date(year = 1990, month = 1, day = 1)
        self.end_date   = dt.date(year = 2026, month = 7, day = 20)
            
    def get_raw_index_data(self, path: str, verbose: bool = True) -> None: 
        
        if verbose: 
            print("Getting Raw Equity Index Data")
        
        out_path    = os.path.join(self.raw_path, "RawEquityIndexData.parquet")
        ticker_path = os.path.join(self.data_path, "EquityForecastTickerGuide.xlsx")
        
        if os.path.exists(out_path) == True: 
            
            if verbose: print("Already have data\n")
            return None
        
        if verbose: 
            print("Combining all the raw data")
        
        idx1_path = os.path.join(path, "EquityIndices1.parquet")
        idx2_path = os.path.join(path, "EquityIndices2.parquet")
        idx3_path = os.path.join(path, "EquityIndices3.parquet")
        
        df_idx1 = (pd
                   .read_parquet(path = idx1_path, engine = "pyarrow")
                   .rename(columns = {"security": "ticker"}))

        df_idx2 = (pd
                .read_parquet(path = idx2_path, engine = "pyarrow")
                .rename(columns = {
                    "variable": "field",
                    "security": "ticker"}))
        
        df_idx3 = (pd
                .read_parquet(path = idx3_path, engine = "pyarrow")
                .rename(columns = {
                    "Date"    : "date",
                    "security": "ticker"})
                .assign(date = lambda x: pd.to_datetime(x.date)).
                dropna())

        df_combined = (pd.concat([
            df_idx1, df_idx2, df_idx3]).
            dropna().
            drop_duplicates())
        
        if verbose: print("Saving data\n")
        df_combined.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_raw_etf_data(self, path: str, verbose: bool = True) -> None: 
        
        if verbose: print("Getting ETF Fundamentals")
        
        in_path  = os.path.join(path, "EquityETFs.parquet")
        out_path = os.path.join(self.raw_path, "RawETFData.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_out = (pd
                .read_parquet(path = in_path, engine = "pyarrow")
                .rename(columns = {"security": "ticker"}))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def collect_equity_rtn(self, verbose: bool = True) -> None: 
        
        if verbose: print("Collecting equity returns")
        
        out_path    = os.path.join(self.raw_path, "RawETFPx.parquet")
        ticker_path = os.path.join(self.data_path, "EquityForecastTickerGuide.xlsx")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        tickers = (pd
                .read_excel(io = ticker_path, sheet_name = "ETFBenchmarkGuide")
                [["Ticker"]]
                .drop_duplicates()
                .assign(ticker = lambda x: x.Ticker.str.split(" ").str[0])
                .ticker
                .to_list())
        
        df_raw = (yf.download(
            tickers     = tickers,
            start       = self.start_date,
            end         = self.end_date,
            auto_adjust = False).
            reset_index().
            melt(id_vars = [("Date", "")]).
            rename(columns = {
                ("Date", ""): "date",
                "Price"     : "px_type",
                "Ticker"    : "ticker"}).
            assign(
                date    = lambda x: pd.to_datetime(x.date),
                px_type = lambda x: x.px_type.str.replace(" ", "").astype(str),
                ticker  = lambda x: x.ticker.astype(str),
                value   = lambda x: x.value.astype(float)))

        if verbose: print("Saving data\n")
        df_raw.to_parquet(path = out_path, engine = "pyarrow")
        
    def seperate_data_by_freq(self, verbose: bool = True) -> None: 
        
        if verbose: print("Seperating Data by Frequency")
        
        file_dict = {
            "index_data": "RawEquityIndexData", 
            "etf_data"  : "RawETFData"}
        
        ticker_path   = os.path.join(self.data_path, "EquityForecastTickerGuide.xlsx")
        df_ticker_map = pd.read_excel(io = ticker_path, sheet_name = "FieldMap")
        
        for key in file_dict.keys():
            
            raw_path = os.path.join(self.raw_path, file_dict[key] + ".parquet")
            df_raw   = pd.read_parquet(path = raw_path, engine = "pyarrow")
            fields   = df_raw.field.drop_duplicates().sort_values().to_list()
            
            frequencys = (df_ticker_map
                    .loc[lambda x: x.field.isin(fields)]
                    .frequency
                    .drop_duplicates()
                    .to_list())
            
            for frequency in frequencys: 

                df_ticker_tmp = (df_ticker_map
                     .loc[lambda x: x.field.isin(fields)]
                     .loc[lambda x: x.frequency == frequency])
                
                groups = df_ticker_tmp.group.drop_duplicates().sort_values().to_list()
                for group in groups: 
        
                    if verbose: print("Working on {} {} {}".format(key, frequency, group))     
                    
                    if   frequency == "daily"    : main_path = self.daily_path
                    elif frequency == "quarterly": main_path = self.quart_path
                
                    out_path = (os
                                .path.join(main_path, "{}{}{}.parquet"
                                           .format(
                                               frequency.capitalize(), 
                                               group.capitalize(),
                                               file_dict[key].replace("Raw", ""))))
                    
                    if os.path.exists(out_path) == True: 
                        if verbose: print("Already have data\n")
                        
                    else: 
                    
                        tmp_fields = (df_ticker_tmp
                                .loc[lambda x: x.group == group]
                                .field
                                .to_list())

                        df_sliced = (df_raw
                                .iloc[lambda x: x.field.isin(tmp_fields)])
                        
                        if verbose: print("Saving data\n")
                        df_sliced.to_parquet(path = out_path, engine = "pyarrow")

def main() -> None:
        
    path         = r"A:\2026BlpAdHocData\Combined\Equity"
    data_collect = DataCollection()
    data_collect.get_raw_index_data(path)
    data_collect.get_raw_etf_data(path)
    data_collect.collect_equity_rtn()
    data_collect.seperate_data_by_freq()
    
if __name__ == "__main__": main()