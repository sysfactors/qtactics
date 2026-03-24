```python
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple
import logging
from functools import reduce
from technical.indicators import cmf

logger = logging.getLogger(__name__)

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class MFI_CMF(IStrategy):

    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short: bool = True

    minimal_roi = {}
    stoploss = -0.20
    trailing_stop = False

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 200

    # ================================
    # ORIGINAL PARAMETERS (UNCHANGED)
    # ================================
    cmf_length = CategoricalParameter([20, 30, 40 ,50], default=50, space="buy", optimize=True)
    cmf_threshold = CategoricalParameter([0.03, 0.05, 0.07], default=0.05, space="buy", optimize=True) 
    mfi_threshold = CategoricalParameter([50, 55, 60, 65], default=55, space="buy")

    risk_ratio = CategoricalParameter([2, 2.5], default=2, space="buy")    
    atr_mult = CategoricalParameter([2, 2.5, 3, 3.5], default=2.5, space="buy")

    leverage_level = IntParameter(1, 10, default=1, space="buy", optimize=False, load=False)

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "ema200": {"color": "#2962FF"}
            },
            "subplots": {
                "MFI": {
                    "mfi": {"color": "#7E57C2"}
                },
                "CMF": {
                    f"cmf{self.cmf_length.value}": {"color": "#43A047"}
                },
                "ADX": {
                    "adx": {"color": "#FF9800"}
                }
            }
        }

    def informative_pairs(self):
        return []

    # ================================
    # INDICATORS
    # ================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe["mfi"] = ta.MFI(dataframe)

        for val in self.cmf_length.range:
            dataframe[f"cmf{val}"] = pta.cmf(
                high=dataframe['high'],
                low=dataframe['low'],
                close=dataframe['close'],
                volume=dataframe['volume'],
                length=val)

        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # 🔥 NEW REGIME FILTERS
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        return dataframe

    # ================================
    # ENTRY
    # ================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                (dataframe['mfi'] > self.mfi_threshold.value) &
                (qtpylib.crossed_above(dataframe[f"cmf{self.cmf_length.value}"], self.cmf_threshold.value)) &
                (dataframe["close"] > dataframe["ema200"]) &

                # 🔥 FILTERS
                (dataframe["adx"] > 20) &
                (dataframe["atr_pct"] > 0.003) &

                (dataframe["volume"] > 0)
            ),
            "enter_long"] = 1
        
        dataframe.loc[
            (
                (dataframe['mfi'] < (100 - self.mfi_threshold.value)) &
                (qtpylib.crossed_below(dataframe[f"cmf{self.cmf_length.value}"], - self.cmf_threshold.value)) &
                (dataframe["close"] < dataframe["ema200"]) &

                # 🔥 FILTERS
                (dataframe["adx"] > 20) &
                (dataframe["atr_pct"] > 0.003) &

                (dataframe["volume"] > 0)
            ),
            "enter_short"] = 1
        
        return dataframe

    # ================================
    # EXIT
    # ================================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                current_profit: float, **kwargs):
        
        side = -1 if trade.is_short else 1

        take_profit = trade.get_custom_data('take_profit')
        stop_loss = trade.get_custom_data('stop_loss')
        
        if take_profit is None or stop_loss is None:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            signal_data = dataframe.loc[dataframe["date"] < trade_date]

            if signal_data.empty:
                return None

            signal_candle = signal_data.iloc[-1]
            
            atr = signal_candle["atr"]
            close = signal_candle["close"]

            take_profit = close + side * self.atr_mult.value * atr * self.risk_ratio.value
            stop_loss = close - side * self.atr_mult.value * atr
            
            trade.set_custom_data('take_profit', take_profit)
            trade.set_custom_data('stop_loss', stop_loss)

        if (trade.is_short and current_rate <= take_profit) or \
           (not trade.is_short and current_rate >= take_profit):
            return "take_profit_achieved"

        if (trade.is_short and current_rate >= stop_loss) or \
           (not trade.is_short and current_rate <= stop_loss):
            return "stop_loss_achieved"

        return None

    # ================================
    # 🔥 FORCE 20x LEVERAGE
    # ================================
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return min(20, max_leverage)
```
