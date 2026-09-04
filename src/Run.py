from IndexEquityCollection import DataCollection
from QuarterlyizeData import Quarterlizer
from PrepData import DataPrep
from BaselineModels import BaselineModels

# Collect Index Equity Data
path         = r"A:\2026BlpAdHocData\Combined\Equity"
data_collect = DataCollection()
data_collect.get_raw_index_data(path)
data_collect.get_raw_etf_data(path)
data_collect.collect_equity_rtn()
data_collect.seperate_data_by_freq()

# Quarterly-ize data
quarterlizer = Quarterlizer()
quarterlizer.quarterlize()
quarterlizer.get_quarterly_returns()

# Get the data and prepare it
data_prep = DataPrep()
data_prep.vol_target_rtn()
data_prep.prep_analyst_est_fundamental()
data_prep.get_date_mapper()

# Start by making baseline models and then comparing them