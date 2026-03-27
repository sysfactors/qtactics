# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple, List
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
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


# ==========================================
# Maximize Crypto Trades: Backtest RSI, MACD & Stochastic with Freqtrade
# YouTube Link: https://youtu.be/pBGvf5wtETg
# ==========================================


# ================================
# Download Historical Data
# ================================

"""
freqtrade download-data \
    -c user_data/binance_futures_RSI_MACD_STOCHASTIC.json \
    --timerange 20230101- \
    -t 1m 5m 15m 30m 1h 2h 4h 1d
"""


# ================================
# Lookahead Analysis 
# ================================

"""
freqtrade lookahead-analysis \
    --strategy RSI_MACD_STOCHASTIC \
    --timeframe 30m \
    --timerange 20231201-20241201 \
    -c user_data/binance_futures_RSI_MACD_STOCHASTIC.json \
    --max-open-trades 1 \
    -p BTC/USDT:USDT
"""

# ================================
# Hyperopt Optimization
# ================================

"""
freqtrade hyperopt \
    --strategy RSI_MACD_STOCHASTIC \
    --config user_data/binance_futures_RSI_MACD_STOCHASTIC.json \
    --timeframe 30m \
    --timerange 20231201-20240801 \
    --hyperopt-loss MultiMetricHyperOptLoss \
    --spaces buy\
    -e 50 \
    --j -2 \
    --random-state 9319 \
    --min-trades 30 \
    --max-open-trades 1 \
    -p BTC/USDT:USDT
"""

# ================================
# Backtesting
# ================================

"""
freqtrade backtesting \
    --strategy RSI_MACD_STOCHASTIC \
    --timeframe 30m \
    --timerange 20231201-20241201 \
    --breakdown month \
    -c user_data/binance_futures_RSI_MACD_STOCHASTIC.json \
    --max-open-trades 1 \
    --cache none \
    --timeframe-detail 5m \
    -p BTC/USDT:USDT
"""

# ================================
# Start FreqUI Web Interface
# ================================

"""
freqtrade webserver \
    --config user_data/binance_futures_RSI_MACD_STOCHASTIC.json
"""


class RSI_MACD_STOCHASTIC(IStrategy):
            
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "30m"

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

    risk_ratio = CategoricalParameter([1.5, 2, 2.5, 3], default=2, space="buy")    
    atr_mult = CategoricalParameter([2, 2.5, 3, 3.5], default=2.5, space="buy")

    stoch_oversold_threshold = CategoricalParameter([20, 25, 30], default=30, space="buy")
    stoch_overbought_threshold = CategoricalParameter([70, 75, 80], default=70, space="buy")
    stoch_rolling_window = CategoricalParameter([5, 10, 15], default=5, space="buy")
    rsi_macd_rolling_window = CategoricalParameter([2, 4, 6], default=2, space="buy")
    rsi_threshold = CategoricalParameter([50, 55, 60], default=50, space="buy")
    
    leverage_level = IntParameter(1, 5, default=1, space='buy', optimize=False, load=False)

    @property
    def plot_config(self):

        plot_config = {
            "main_plot": {
            },
            "subplots": {
                "STOCH": {
                        'fastk': {'color': '#2962ff'},
                        'fastd': {'color': '#ff6d00'}
                    },
                "RSI": {
                    f"rsi": {
                        "color": "#9e57c2",
                        "type": "line"
                    }
                },
                "MACD": {
                    'macd': {'color': '#2962ff', 'fill_to': 'macdhist'},
                    'macdsignal': {'color': '#ff6d00'},
                    'macdhist': {'type': 'bar', 'plotly': {'opacity': 0.9}}
                    }
            }
        }
        
        return plot_config
    
    
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
        # pairs = self.dp.current_whitelist()

        # # Assign tf to each pair so they can be downloaded and cached for strategy.
        # informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        
        return []
    
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # Stochastic Fast with parameters 14, 3, 3
        stoch_fast = ta.STOCHF(dataframe, fastk_period=14, fastd_period=3, fastd_matype=3)
        dataframe["fastd"] = stoch_fast["fastd"]
        dataframe["fastk"] = stoch_fast["fastk"]
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]
        
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)

        # ATR
        dataframe["atr"] = ta.ATR(dataframe)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:        
        
        dataframe.loc[
            (   
                # Condition 1: Stochastic oversold and overbought
                (dataframe['fastd'].rolling(window=self.stoch_rolling_window.value).min() < self.stoch_oversold_threshold.value) &
                (dataframe['fastk'].rolling(window=self.stoch_rolling_window.value).min() < self.stoch_oversold_threshold.value) &
                (dataframe['fastd'] > self.stoch_oversold_threshold.value) &
                (dataframe['fastk'] > self.stoch_oversold_threshold.value) &
                (dataframe['fastd'] < self.stoch_overbought_threshold.value) &
                (dataframe['fastk'] < self.stoch_overbought_threshold.value) &
                
                # Condition 2: RSI cross over
                (dataframe['rsi'] > self.rsi_threshold.value) &
                (dataframe['rsi'].rolling(window=self.rsi_macd_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_above(x, self.rsi_threshold.value))
                )) & 
                
                # Condition 3: MACD crossover
                (dataframe['macd'] > dataframe['macdsignal']) &
                (dataframe['macd'].rolling(window=self.rsi_macd_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_above(x, dataframe['macdsignal'].iloc[x.index[0]:x.index[-1]+1])) 
                ))

            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                # (dataframe['close'] < dataframe['ema200']) &
                
                # Condition 1: Stochastic overbought and oversold
                (dataframe['fastd'].rolling(window=self.stoch_rolling_window.value).max() > self.stoch_overbought_threshold.value) &
                (dataframe['fastk'].rolling(window=self.stoch_rolling_window.value).max() > self.stoch_overbought_threshold.value) &
                (dataframe['fastd'] < self.stoch_overbought_threshold.value) &
                (dataframe['fastk'] < self.stoch_overbought_threshold.value) &
                (dataframe['fastd'] > self.stoch_oversold_threshold.value) &
                (dataframe['fastk'] > self.stoch_oversold_threshold.value) &
                
                # Condition 2: RSI cross below 50
                (dataframe['rsi'] < self.rsi_threshold.value) &
                (dataframe['rsi'].rolling(window=self.rsi_macd_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_below(x, 100 - self.rsi_threshold.value))
                )) &
                
                # Condition 3: MACD crossover
                (dataframe['macd'] < dataframe['macdsignal']) &
                (dataframe['macd'].rolling(window=self.rsi_macd_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_below(x, dataframe['macdsignal'].iloc[x.index[0]:x.index[-1]+1])) 
                ))
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

        # Retrieve TP and SL from custom data
        take_profit = trade.get_custom_data('take_profit')
        stop_loss = trade.get_custom_data('stop_loss')
        
        # If TP or SL is not set, initialize them
        if take_profit is None or stop_loss is None:
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

            take_profit = close + side * self.atr_mult.value * atr * self.risk_ratio.value
            stop_loss = close - side * self.atr_mult.value * atr
            
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
            # logger.info(f"[{pair}] Take Profit hit! Close: {current_close:.2f}, TP: {take_profit:.2f}")
            return "take_profit_achieved"

        if (trade.is_short and current_close >= stop_loss) or \
        (not trade.is_short and current_close <= stop_loss):
            # logger.info(f"[{pair}] Stop Loss hit! Close: {current_close:.2f}, SL: {stop_loss:.2f}")
            return "stop_loss_achieved"

        return None
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_level.value