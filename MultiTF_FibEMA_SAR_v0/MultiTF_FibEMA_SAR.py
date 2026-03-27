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


# ================================
# Freqtrade Version
# ================================

"""
freqtrade -V
Freqtrade Version: freqtrade 2025.1
"""

# ================================
# Download Historical Data
# ================================

"""
freqtrade download-data \
    -c user_data/binance_futures_MultiTF_FibEMA_SAR.json \
    --timerange 20230101- \
    -t 1m 5m 15m 30m 1h 4h 1d
"""

# ================================
# Lookahead Analysis 
# ================================

"""
freqtrade lookahead-analysis \
    --strategy MultiTF_FibEMA_SAR \
    --timeframe 1h \
    --timerange 20240101-20250101 \
    -c user_data/binance_futures_MultiTF_FibEMA_SAR.json \
    --max-open-trades 1 \
    -p DOT/USDT:USDT
"""

# ================================
# Hyperopt Optimization
# ================================

"""
freqtrade hyperopt \
    --strategy MultiTF_FibEMA_SAR \
    --config user_data/binance_futures_MultiTF_FibEMA_SAR.json \
    --timeframe 1h \
    --timerange 20240101-20240901 \
    --hyperopt-loss SharpeHyperOptLossDaily \
    --spaces buy\
    -e 55 \
    --j -2 \
    --random-state 9319 \
    --min-trades 20 \
    --max-open-trades 1 \
    -p DOT/USDT:USDT
"""

# ================================
# Backtesting
# ================================

"""
freqtrade backtesting \
    --strategy MultiTF_FibEMA_SAR \
    --timeframe 1h \
    --timerange 20240101-20250101 \
    --breakdown month \
    -c user_data/binance_futures_MultiTF_FibEMA_SAR.json \
    --max-open-trades 1 \
    --cache none \
    --timeframe-detail 1m \
    -p DOT/USDT:USDT
"""

# ================================
# Start FreqUI Web Interface
# ================================

"""
freqtrade webserver \
    --config user_data/config_binance_futures.json
"""

class MultiTF_FibEMA_SAR(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "1h"

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {}
    
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.25

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
    ema_set = CategoricalParameter(["3_5_8", "5_8_13", "8_13_21", "13_21_34"], default="5_8_13", space='buy')
    ema_cross_rolling_window = IntParameter(1, 8, default=4, space='buy')
    stop_loss_buffer = DecimalParameter(0.001, 0.008, decimals=3, default=0.005, space="buy")  
    swing_point_lookback = IntParameter(3, 6, default=3, space='buy')
    risk_ratio = CategoricalParameter([1.5, 2, 2.5, 3], default=2, space="buy")  

    leverage_level = IntParameter(1, 10, default=1, space='buy', optimize=False, load=False)
    
    @property
    def plot_config(self):
        
        short_val, mid_val, long_val = [int(x) for x in self.ema_set.value.split("_")]  
        
        return {
            "main_plot": {
                f"ema{short_val}": {"color": "red"},
                f"ema{mid_val}": {"color": "blue"},
                f"ema{long_val}": {"color": "green"},
                f"sar": {
                    "color": "#e0a217",
                    "type": "scatter",
                    "scatterSymbolSize": 3
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
        
        for ema_set_val in self.ema_set.range:        
            short_val, mid_val, long_val = [int(x) for x in ema_set_val.split('_')]   
        
            dataframe[f"ema{short_val}"] = ta.EMA(dataframe, timeperiod=short_val)
            dataframe[f"ema{mid_val}"] = ta.EMA(dataframe, timeperiod=mid_val)
            dataframe[f"ema{long_val}"] = ta.EMA(dataframe, timeperiod=long_val)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # EMA
        for ema_set_val in self.ema_set.range:        
            short_val, mid_val, long_val = [int(x) for x in ema_set_val.split("_")]   
        
            dataframe[f"ema{short_val}"] = ta.EMA(dataframe, timeperiod=short_val)
            dataframe[f"ema{mid_val}"] = ta.EMA(dataframe, timeperiod=mid_val)
            dataframe[f"ema{long_val}"] = ta.EMA(dataframe, timeperiod=long_val)
        
        # SAR
        dataframe["sar"] = ta.SAR(dataframe)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        short_val, mid_val, long_val = [int(x) for x in self.ema_set.value.split("_")]   

        dataframe.loc[
            (
                # EMA values on 4H timeframe (short > mid > long)
                (dataframe[f"ema{short_val}_4h"] > dataframe[f"ema{mid_val}_4h"]) &
                (dataframe[f"ema{mid_val}_4h"] > dataframe[f"ema{long_val}_4h"]) &

                # EMA values on the current timeframe (short > mid > long)
                (dataframe[f"ema{short_val}"] > dataframe[f"ema{mid_val}"]) &
                (dataframe[f"ema{mid_val}"] > dataframe[f"ema{long_val}"]) &

                # Check if EMA(short) crossed above EMA(long) in the rolling window
                (dataframe[f"ema{short_val}"].rolling(window=self.ema_cross_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_above(x, dataframe[f"ema{long_val}"].iloc[x.index[0]:x.index[-1]+1])))) &
                # Check if EMA(mid) crossed above EMA(long) in the rolling window
                (dataframe[f"ema{mid_val}"].rolling(window=self.ema_cross_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_above(x, dataframe[f"ema{long_val}"].iloc[x.index[0]:x.index[-1]+1])))) &

                # Close price is above the Parabolic SAR (SAR)
                (dataframe["close"] > dataframe["sar"]) &

                # Volume is greater than 0 (indicating valid trading activity)
                (dataframe["volume"] > 0)
            ),
            "enter_long"] = 1

        dataframe.loc[
            (
                # EMA values on 4H timeframe (short < mid < long)
                (dataframe[f"ema{short_val}_4h"] < dataframe[f"ema{mid_val}_4h"]) &
                (dataframe[f"ema{mid_val}_4h"] < dataframe[f"ema{long_val}_4h"]) &

                # EMA values on the current timeframe (short < mid < long)
                (dataframe[f"ema{short_val}"] < dataframe[f"ema{mid_val}"]) &
                (dataframe[f"ema{mid_val}"] < dataframe[f"ema{long_val}"]) &

                # Check if EMA(short) crossed below EMA(long) in the rolling window
                (dataframe[f"ema{short_val}"].rolling(window=self.ema_cross_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_below(x, dataframe[f"ema{long_val}"].iloc[x.index[0]:x.index[-1]+1])))) &
                # Check if EMA(mid) crossed below EMA(long) in the rolling window
                (dataframe[f"ema{mid_val}"].rolling(window=self.ema_cross_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_below(x, dataframe[f"ema{long_val}"].iloc[x.index[0]:x.index[-1]+1])))) &

                # Close price is below the Parabolic SAR (SAR)
                (dataframe["close"] < dataframe["sar"]) &

                # Volume is greater than 0 (indicating valid trading activity)
                (dataframe["volume"] > 0)
            ),
            "enter_short"] = 1 

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0
        
        return dataframe
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
        current_profit: float, **kwargs):

        # Retrieve TP and SL from custom data
        take_profit = trade.get_custom_data('take_profit')
        stop_loss = trade.get_custom_data('stop_loss')
        
        # If TP or SL is not set, initialize them
        if take_profit is None or stop_loss is None:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

            # Get the date just before trade opened
            trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            trade_signal = dataframe.loc[dataframe["date"] < trade_date]

            # Filter dataframe to candles before the trade opened
            signal_data = dataframe.loc[dataframe["date"] < trade_date]

            if signal_data.empty:
                logger.warning(f"[{pair}] No signal candle found. Skip setting TP/SL.")
                return None

            signal_candle = signal_data.iloc[-1]
            
            if not trade.is_short:
                # Calculate the lowest price over the last x candles (swing low)
                swing_low = trade_signal["low"].rolling(self.swing_point_lookback.value).min().iloc[-1]
                stop_loss = swing_low * (1 - self.stop_loss_buffer.value)
                risk_amount = abs(signal_candle["close"] - stop_loss)
                take_profit = signal_candle["close"] + (self.risk_ratio.value * risk_amount)
                
            elif trade.is_short:
                # Calculate the highest price over the last x candles (swing high)
                swing_high = trade_signal["high"].rolling(self.swing_point_lookback.value).max().iloc[-1]
                stop_loss = swing_high * (1 + self.stop_loss_buffer.value)
                risk_amount = abs(signal_candle["close"] - stop_loss)
                take_profit = signal_candle["close"] - (self.risk_ratio.value * risk_amount)
            
            # Save to trade's custom data
            trade.set_custom_data('take_profit', take_profit)
            trade.set_custom_data('stop_loss', stop_loss)

            # logger.info(f"[{pair}] TP/SL set. TP: {take_profit:.2f}, SL: {stop_loss:.2f}")
            
        # Get the current close price
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_close = dataframe.iloc[-1]['close']

        # Check exit conditions
        if (trade.is_short and current_close <= take_profit) or \
        (not trade.is_short and current_close >= take_profit):
            logger.info(f"[{pair}] Take Profit hit! Close: {current_close:.2f}, TP: {take_profit:.2f}")
            return "take_profit_achieved"

        if (trade.is_short and current_close >= stop_loss) or \
        (not trade.is_short and current_close <= stop_loss):
            logger.info(f"[{pair}] Stop Loss hit! Close: {current_close:.2f}, SL: {stop_loss:.2f}")
            return "stop_loss_achieved"

        return None
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_level.value