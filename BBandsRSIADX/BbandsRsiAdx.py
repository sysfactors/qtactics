# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple
import logging
from functools import reduce

logger = logging.getLogger(__name__)

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib

# ==========================================
# Mean Reversion Trading Strategy Explained & Backtested – 179% Profit
# https://youtu.be/c9-SIpy3dEw
# ==========================================

# ================================
# Freqtrade Version
# ================================

"""
freqtrade -V

Operating System:       Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.36
Python Version:         Python 3.12.10
CCXT Version:           4.4.82

Freqtrade Version:      freqtrade 2025.5
"""

# ================================
# Download Historical Data
# ================================

"""
freqtrade download-data \
    -c user_data/binance_futures_BbandsRsiAdx.json \
    --timerange 20230101- \
    -t 5m 15m 30m 1h 2h 4h 1d
"""

# ================================
# Hyperopt Optimization
# ================================

"""
freqtrade hyperopt \
    --strategy BbandsRsiAdx \
    --config user_data/binance_futures_BbandsRsiAdx.json \
    --timeframe 1h \
    --timerange 20240601-20250201 \
    --hyperopt-loss MultiMetricHyperOptLoss \
    --spaces buy\
    -e 50 \
    --j -2 \
    --random-state 9319 \
    --min-trades 20 \
    --max-open-trades 1 \
    -p NEAR/USDT:USDT
"""

# ================================
# Backtesting
# ================================

"""
freqtrade backtesting \
    --strategy BbandsRsiAdx \
    --timeframe 1h \
    --timerange 20240601-20250601 \
    --breakdown month \
    -c user_data/binance_futures_BbandsRsiAdx.json \
    --max-open-trades 1 \
    --timeframe-detail 5m \
    -p NEAR/USDT:USDT \
    --cache none
"""

# ================================
# Start FreqUI Web Interface
# ================================

"""
freqtrade webserver \
    --config user_data/config_binance_futures.json
"""


    
class BbandsRsiAdx(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "1h"
    informative_timeframe = '4h'

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {}
    
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.20

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200
    
    # Strategy parameters
    bb_period = CategoricalParameter([15, 20], default=20, space="buy")    
    bb_stds = CategoricalParameter([1.5, 2], default=2, space="buy")   
    htf_rsi_threshold = CategoricalParameter([50, 55], default=50, space="buy") 
    htf_adx_threshold = CategoricalParameter([25, 30], default=30, space="buy") 
    adx_threshold = CategoricalParameter([20, 25], default=30, space="buy") 
    
    atr_mult = CategoricalParameter([3, 3.5, 4, 4.5, 5], default=2, space="buy")    
    
    leverage_level = IntParameter(1, 3, default=1, space='buy', optimize=False, load=False)
    
    @property
    def plot_config(self):
        return {
            "main_plot": {
                f"bb_lowerband_{self.bb_period.value}_{self.bb_stds.value}": {"color": "white"},
                f"bb_upperband_{self.bb_period.value}_{self.bb_stds.value}": {"color": "white"},
            },
            "subplots": {
                
                "RSI_4h": {
                    "rsi_4h": {"color": "#9e57c2", "type": "line"
                    }
                },
                "ADX": {
                    "adx": {"color": "#9e57c2", "type": "line"
                    }
                },
                "ADX_4H": {
                    "adx_4h": {"color": "#9e57c2", "type": "line"
                    }
                }
            }
        }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()

        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        
        return informative_pairs
    
    
    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)
        
        # ADX
        dataframe["adx"] = ta.ADX(dataframe)

        return dataframe
    

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        for bb_period_val in self.bb_period.range:       
            for bb_stds_val in self.bb_stds.range: 
                bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=bb_period_val, stds=bb_stds_val)
                dataframe[f"bb_lowerband_{bb_period_val}_{bb_stds_val}"] = bollinger["lower"]
                dataframe[f"bb_middleband_{bb_period_val}_{bb_stds_val}"] = bollinger["mid"]
                dataframe[f"bb_upperband_{bb_period_val}_{bb_stds_val}"] = bollinger["upper"]

                dataframe[f"bb_width_{bb_period_val}_{bb_stds_val}"] = (
                    (dataframe[f"bb_upperband_{bb_period_val}_{bb_stds_val}"] - dataframe[f"bb_lowerband_{bb_period_val}_{bb_stds_val}"]) / dataframe[f"bb_middleband_{bb_period_val}_{bb_stds_val}"]
                )
        
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)
        
        # ADX
        dataframe["adx"] = ta.ADX(dataframe)

        # ATR
        dataframe["atr"] = ta.ATR(dataframe)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
                (
                    (qtpylib.crossed_below(dataframe["close"], dataframe[f"bb_lowerband_{self.bb_period.value}_{self.bb_stds.value}"])) &
                    (dataframe["rsi_4h"] > self.htf_rsi_threshold.value) &
                    (dataframe["adx_4h"] > self.htf_adx_threshold.value) &
                    (dataframe["adx"] > self.adx_threshold.value) &
                    (dataframe["volume"] > 0)
                ),
                'enter_long'] = 1
        
        dataframe.loc[
                (
                    (qtpylib.crossed_above(dataframe["close"], dataframe[f"bb_upperband_{self.bb_period.value}_{self.bb_stds.value}"])) &
                    (dataframe["rsi_4h"] < (100 - self.htf_rsi_threshold.value)) &
                    (dataframe["adx_4h"] > self.htf_adx_threshold.value) &
                    (dataframe["adx"] > self.adx_threshold.value) &
                    (dataframe["volume"] > 0)
                ),
                'enter_short'] = 1

        return dataframe
    

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["close"], dataframe[f"bb_upperband_{self.bb_period.value}_{self.bb_stds.value}"])) &
                (dataframe["volume"] > 0)
            ),
            "exit_long"] = 1
        
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe["close"], dataframe[f"bb_lowerband_{self.bb_period.value}_{self.bb_stds.value}"])) &
                (dataframe["volume"] > 0)
            ),
            "exit_short"] = 1

        return dataframe
    

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                current_profit: float, **kwargs):
        
        side = -1 if trade.is_short else 1

        # Retrieve TP and SL from custom data
        # take_profit = trade.get_custom_data('take_profit')
        stop_loss = trade.get_custom_data('stop_loss')
        
        # If TP or SL is not set, initialize them
        if stop_loss is None:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

            # Get the date just before trade opened
            trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)

            # Filter dataframe to candles before the trade opened
            signal_data = dataframe.loc[dataframe["date"] < trade_date]

            if signal_data.empty:
                logger.warning(f"[{pair}] No signal candle found. Skip setting TP/SL.")
                return None

            signal_candle = signal_data.iloc[-1]
            
            # Calculate TP and SL
            atr = signal_candle["atr"]
            close = signal_candle["close"]

            # take_profit = close + side * self.atr_mult.value * atr * self.risk_ratio.value
            stop_loss = close - side * self.atr_mult.value * atr
            
            # Save to trade's custom data
            # trade.set_custom_data('take_profit', take_profit)
            trade.set_custom_data('stop_loss', stop_loss)

            # logger.info(f"[{pair}] TP/SL set. TP: {take_profit:.2f}, SL: {stop_loss:.2f}")

        # Check exit conditions
        # if (trade.is_short and current_rate <= take_profit) or \
        # (not trade.is_short and current_rate >= take_profit):
        #     # logger.info(f"[{pair}] Take Profit hit! Close: {current_close:.2f}, TP: {take_profit:.2f}")
        #     return "take_profit_achieved"

        if (trade.is_short and current_rate >= stop_loss) or \
        (not trade.is_short and current_rate <= stop_loss):
            # logger.info(f"[{pair}] Stop Loss hit! Close: {current_close:.2f}, SL: {stop_loss:.2f}")
            return "stop_loss_achieved"

        return None
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_level.value