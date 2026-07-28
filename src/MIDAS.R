library(TTR)
library(arrow)
library(dplyr)
library(purrr)
library(midasr)
library(lubridate)
library(tidyverse)

src_cwd    <- getwd()
repo_path  <- dirname(src_cwd)
data_path  <- file.path(repo_path, "data")
base_path  <- file.path(data_path, "Models", "BaselineModels")
db_path    <- file.path(data_path, "DataBase") 
daily_path <- file.path(db_path, "DailyData")

sample_tickers <- c("SPX Index", "NDX Index", "RTY Index")

path <- file.path(db_path, "Quarterlyized", "QuarterlyFundamentalEquityIndexData.parquet")

df_fundamental <- read_parquet(file = path) %>% 
  filter(ticker %in% sample_tickers) %>% 
  filter(stat_method == "last") %>% 
  select(quarter, value, ticker, field) %>% 
  group_by(ticker, field) %>% 
  arrange(quarter) %>% 
  mutate(
    diff_val = value - lag(value),
    diff_val_lag = lag(diff_val)
  ) %>% 
  drop_na() %>% 
  ungroup()

path   <- file.path(base_path, "VolTargetedReturns.parquet")
df_rtn <- read_parquet(file = path) %>% 
  filter(ticker %in% sample_tickers) %>% 
  arrange(ticker, date) %>% 
  group_by(ticker) %>% 
  mutate(
    quarter = paste0(year(date), "-Q", quarter(date)),
    rtn_lag = lag(rtn_val)
  ) %>% 
  ungroup()

df_fundamental_tmp <- df_fundamental %>% 
  filter(ticker == min(ticker)) %>% 
  filter(field == min(field)) %>% 
  arrange(quarter) 
  
start_quarter <- df_fundamental_tmp %>% 
  select(quarter) %>% 
  filter(quarter == min(quarter)) %>% 
  pull()

end_quarter <- df_fundamental_tmp %>% 
  select(quarter) %>% 
  filter(quarter == max(quarter)) %>% 
  pull()

rtn <- df_rtn %>% 
  mutate(quarter = paste0(year(date), "-Q", quarter(date))) %>% 
  filter(ticker == min(ticker)) %>% 
  filter(vol_target == min(vol_target)) %>% 
  filter(start_quarter < quarter) %>% 
  filter(quarter < end_quarter) %>% 
  arrange(date) %>% 
  select(rtn_lag) %>% 
  pull()

df_tmp_rtn <- df_rtn %>% 
  mutate(quarter = paste0(year(date), "-Q", quarter(date))) %>% 
  filter(ticker == min(ticker)) %>% 
  filter(vol_target == min(vol_target)) %>% 
  filter(start_quarter < quarter) %>% 
  filter(quarter < end_quarter)

min_count <- df_tmp_rtn %>% 
  count(quarter) %>% 
  pull(n) %>% 
  min()

rtn_lag <- df_tmp_rtn  %>% 
  group_by(quarter) %>% 
  slice_head(n = min_count) %>% 
  ungroup() %>% 
  select(rtn_lag) %>% 
  pull()

fund_vals <- df_fundamental_tmp %>% 
  filter(start_quarter < quarter) %>% 
  filter(quarter < end_quarter) %>% 
  arrange(quarter) %>% 
  select(diff_val) %>% 
  pull()

model1 <- lm(fund_vals ~ mls(rtn_lag, k = 0:(min_count-1), m = min_count))
model2 <- midas_u(fund_vals ~ mls(rtn_lag, k = 0:(min_count-1), m = min_count))

model3 <- midas_r(
  formula = fund_vals ~ mls(rtn_lag, 0:(min_count-1), min_count, nealmon), 
  start   = list(rtn_lag = c(1, -0.5)))



