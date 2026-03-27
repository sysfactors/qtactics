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
# Build a Crypto Trading Bot with Stochastic RSI & EMA: 207% Profit in 1 Year (Python Code Included)
# https://youtu.be/wEeUY-nQyrw
# ==========================================

# ================================
# Freqtrade Version
# ================================

"""
freqtrade -V

Operating System:       Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.36
Python Version:         Python 3.12.9
CCXT Version:           4.4.69

Freqtrade Version:      freqtrade 2025.3
"""

# ================================
# Download Historical Data
# ================================

"""
freqtrade download-data \
    -c user_data/binance_futures_STOCHRSI_EMA.json \
    --timerange 20230101- \
    -t 1m 5m 15m 30m 1h 2h 4h 1d
"""

# ================================
# Hyperopt Optimization
# ================================

"""
freqtrade hyperopt \
    --strategy STOCHRSI_EMA \
    --config user_data/binance_futures_STOCHRSI_EMA.json \
    --timeframe 1h \
    --timerange 20240301-20241101 \
    --hyperopt-loss MultiMetricHyperOptLoss \
    --spaces buy\
    -e 55 \
    --j -2 \
    --random-state 9319 \
    --min-trades 20 \
    --max-open-trades 1 \
    -p TON/USDT:USDT
"""

# ================================
# Backtesting
# ================================

"""
freqtrade backtesting \
    --strategy STOCHRSI_EMA \
    --timeframe 1h \
    --timerange 20240301-20250301 \
    --breakdown month \
    -c user_data/binance_futures_STOCHRSI_EMA.json \
    --max-open-trades 1 \
    --timeframe-detail 5m \
    -p TON/USDT:USDT \
    --cache none
"""

# ================================
# Start FreqUI Web Interface
# ================================

"""
freqtrade webserver \
    --config user_data/binance_futures_STOCHRSI_EMA.json
"""

class STOCHRSI_EMA(IStrategy):

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
    
    # Dictionary defining the exit points for take profit and stop loss levels.
    exit_loss_profit = {}
    
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
    ema_slope_window = CategoricalParameter([5, 10, 15], default=10, space='buy')
    
    stochrsi_oversold_threshold = CategoricalParameter([20, 25, 30], default=25, space="buy")
    stochrsi_overbought_threshold = CategoricalParameter([70, 75, 80], default=75, space="buy")
    stochrsi_threshold_window = CategoricalParameter([3, 6, 9], default=6, space="buy")

    swing_point_lookback =  CategoricalParameter([3, 6, 9, 12], default=6, space='buy')
    risk_ratio = CategoricalParameter([1.5, 2, 2.5], default=2, space="buy")
    
    leverage_level = IntParameter(1, 10, default=1, space='buy', optimize=False, load=False)
    
    @property
    def plot_config(self):
        return {
            "main_plot": {
                "ema50": {
                    "color": "#2962ff",
                    "type": "line"
                },
                "ema100": {
                    "color": "#ffeb3b",
                    "type": "line"
                }
            },
            "subplots": {
                "STOCHRSI": {
                    "srsi_k": {"color": "#2962ff"},
                    "srsi_d": {"color": "#ff6d00"}
                }
            }
        }

    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        #StochRSI 
        period = 14 
        smoothD = 3
        SmoothK = 3
        
        stochrsi  = (dataframe['rsi'] - dataframe['rsi'].rolling(period).min()) / (dataframe['rsi'].rolling(period).max() - dataframe['rsi'].rolling(period).min())
        dataframe['srsi_k'] = stochrsi.rolling(SmoothK).mean() * 100
        dataframe['srsi_d'] = dataframe['srsi_k'].rolling(smoothD).mean()
            
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        ema50_slope = (dataframe["ema50"] - dataframe["ema50"].shift(1)).rolling(window=self.ema_slope_window.value).mean()
        ema100_slope = (dataframe["ema100"] - dataframe["ema100"].shift(1)).rolling(window=self.ema_slope_window.value).mean()

        ema_gap_now = dataframe["ema50"] - dataframe["ema100"]
        ema_gap_before = ema_gap_now.shift(1)
        
        # Long Entry Conditions
        dataframe.loc[
            (
                # EMA Trend Condition: 50-EMA above 100-EMA
                (dataframe["ema50"] > dataframe["ema100"]) &  
                # Price Position: Close price above 50-EMA
                (dataframe["close"] > dataframe["ema50"]) &
                
                # EMA Slope Conditions: Both EMAs sloping upward
                (ema50_slope > 0) &
                (ema100_slope > 0) &
                # EMA Gap Expansion: Current EMA gap wider than previous
                (ema_gap_now > ema_gap_before) &
                
                # Stochastic RSI Conditions:
                # K-line just crossed above oversold threshold
                (qtpylib.crossed_above(dataframe["srsi_k"], self.stochrsi_oversold_threshold.value)) &
                # K-line was below oversold threshold for the entire lookback window
                (dataframe["srsi_k"].shift(1).rolling(window=self.stochrsi_threshold_window.value).apply(lambda x: all(x < self.stochrsi_oversold_threshold.value))) &

                # Volume Filter: Ensure there's trading activity
                (dataframe['volume'] > 0) 
            ),
            'enter_long'] = 1 
                
        # Short Entry Conditions
        dataframe.loc[
            (
                # EMA Trend Condition: 50-EMA below 100-EMA
                (dataframe["ema50"] < dataframe["ema100"]) &  
                
                # EMA Slope Conditions: Both EMAs sloping downward
                (ema50_slope < 0) & 
                (ema100_slope < 0) & 
                # EMA Gap Contraction: Current EMA gap narrower than previous
                (ema_gap_now < ema_gap_before) & 
                
                # Price Position: Close price below 50-EMA
                (dataframe["close"] < dataframe["ema50"]) & 
                
                # Stochastic RSI Conditions:
                # K-line just crossed below overbought threshold
                (qtpylib.crossed_below(dataframe["srsi_k"], self.stochrsi_overbought_threshold.value)) &
                # K-line was above overbought threshold for the entire lookback window
                (dataframe["srsi_k"].shift(1).rolling(window=self.stochrsi_threshold_window.value).apply(lambda x: all(x > self.stochrsi_overbought_threshold.value))) &

                # Volume Filter: Ensure there's trading activity
                (dataframe['volume'] > 0)
            ),
            'enter_short'] = 1

        return dataframe
    

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0

        return dataframe
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        
        side = -1 if trade.is_short else 1
        
        # Check if exit conditions for the pair have been defined
        if pair not in self.exit_loss_profit:
            
            # Initialize exit conditions (take profit and stop loss) for the pair if not already set
            # If signal candle data is not available or incorrect, use default settings with fixed 5% take profit and stop loss
            self.exit_loss_profit[pair] = {
                "take_profit": trade.open_rate * (1 + side * 0.05),
                "stop_loss": trade.open_rate * (1 - side * 0.05),
            }
            
            # Retrieve the analyzed dataframe for the pair and timeframe, then get the historical data prior to the trade's open date.
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            trade_signal = dataframe.loc[dataframe["date"] < trade_date]
            
            # Ensure `trade_signal` is not empty before accessing the last row
            if not trade_signal.empty:   
                signal_candle = trade_signal.iloc[-1].squeeze()
                
                if not trade.is_short:
                    # Calculate the lowest price over the last x candles (swing low)
                    swing_low = trade_signal["low"].rolling(self.swing_point_lookback.value).min().iloc[-1]
                    stop_loss = swing_low * (1 - 0.005)
                    risk_amount = abs(signal_candle["close"] - stop_loss)
                    take_profit = signal_candle["close"] + (self.risk_ratio.value * risk_amount)
                    
                elif trade.is_short:
                    # Calculate the highest price over the last x candles (swing high)
                    swing_high = trade_signal["high"].rolling(self.swing_point_lookback.value).max().iloc[-1]
                    stop_loss = swing_high * (1 + 0.005)
                    risk_amount = abs(signal_candle["close"] - stop_loss)
                    take_profit = signal_candle["close"] - (self.risk_ratio.value * risk_amount)
                
                # Update take profit and stop loss
                self.exit_loss_profit[pair]["take_profit"] = take_profit
                self.exit_loss_profit[pair]["stop_loss"] = stop_loss

            else:
                # Log if no signal candle data is available and default values are used
                logger.warning(f"No signal candle found for {pair}. Using default take profit and stop loss: 5%")
            
        # Get the most recent candle data
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if (trade.is_short and current_candle["close"] <= self.exit_loss_profit[pair]["take_profit"]) or \
            (not trade.is_short and current_candle["close"] >= self.exit_loss_profit[pair]["take_profit"]):
            return "take_profit_achieved"

        if (trade.is_short and current_candle["close"] >= self.exit_loss_profit[pair]["stop_loss"]) or \
            (not trade.is_short and current_candle["close"] <= self.exit_loss_profit[pair]["stop_loss"]):
            return "stop_loss_achieved"

            
    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        
        # Confirms the trade exit by removing the exit loss/profit levels for the given pair
        if pair in self.exit_loss_profit:
            del self.exit_loss_profit[pair]
            
        return True
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_level.value