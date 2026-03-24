import math
import json
from typing import Dict, Any, Union

def calculate_max_dca_stake_final(
    strategy_params: Union[str, Dict[str, Any]],
    tradable_balance_ratio: float = 1.0
) -> float:
    """
    Calculate the theoretical max capital needed for one DCA trade (base + safety orders),
    including detailed price deviations and order sizes according to current parameter design.

    Parameters:
        strategy_params: dict or JSON string containing strategy params (including base_order_size,
                         safety_order_ratio, safety_order_volume_scale, etc.)
        tradable_balance_ratio: ratio of wallet balance allowed per trade (default 1.0)

    Returns:
        float: total wallet balance required to safely run a single trade with all SOs.
    """

    if isinstance(strategy_params, str):
        try:
            params = json.loads(strategy_params)
        except json.JSONDecodeError:
            raise ValueError("Input string is not valid JSON.")
    elif isinstance(strategy_params, dict):
        params = strategy_params
    else:
        raise TypeError("Input must be JSON string or dict.")

    try:
        buy_params = params['params']['buy']

        base_order_size = buy_params['base_order_size']
        safety_order_ratio = buy_params.get('safety_order_ratio', 2.0)
        safety_order_volume_scale = buy_params['safety_order_volume_scale']
        safety_order_max_count = buy_params['safety_order_max_count']
        price_deviation_initial = buy_params['price_deviation_initial']
        safety_order_step_scale = buy_params['safety_order_step_scale']

        take_profit = buy_params.get('take_profit', 0.0)

        if 'tradable_balance_ratio' in params.get('params', {}):
            ratio_obj = params['params']['tradable_balance_ratio']
            if isinstance(ratio_obj, dict) and 'tradable_balance_ratio' in ratio_obj:
                tradable_balance_ratio = ratio_obj['tradable_balance_ratio']
            elif isinstance(ratio_obj, float):
                tradable_balance_ratio = ratio_obj

    except KeyError as e:
        raise KeyError(f"Missing required parameter {e} in 'params'->'buy'.")

    detailed_results = [{
        'Order': 'BO',
        'Level': 0,
        'Deviation': 0.0,
        'Amount': base_order_size,
        'Cumulative_Amount': base_order_size
    }]

    total_so_amount = 0.0
    cumulative_amount = base_order_size

    for i in range(safety_order_max_count):
        so_level = i + 1  

        so_amount = base_order_size * safety_order_ratio * (safety_order_volume_scale ** i)
        total_so_amount += so_amount
        cumulative_amount += so_amount

        if safety_order_step_scale == 1:
            deviation_from_bo = - price_deviation_initial * so_level
        else:
            deviation_from_bo = - price_deviation_initial * ((safety_order_step_scale ** so_level - 1) / (safety_order_step_scale - 1))

        detailed_results.append({
            'Order': f'SO{so_level}',
            'Level': so_level,
            'Deviation': deviation_from_bo,
            'Amount': so_amount,
            'Cumulative_Amount': cumulative_amount
        })

    total_theoretical_stake = base_order_size + total_so_amount

    if tradable_balance_ratio <= 0 or tradable_balance_ratio > 1:
        raise ValueError("tradable_balance_ratio must be between 0 and 1.")

    real_wallet_required = total_theoretical_stake / tradable_balance_ratio

    max_cumulative_dev = detailed_results[-1]['Deviation']

    print("=" * 85)
    print("💰 DCA FUNDING & DEVIATION ANALYSIS (UPDATED VERSION)")
    print("=" * 85)
    print("--- Strategy Parameters ---")
    print(f"| Base Order Size: {base_order_size:.2f} USDT")
    print(f"| Safety Order Ratio (SO1 / BO): {safety_order_ratio:.2f}")
    print(f"| Safety Order Volume Scale: {safety_order_volume_scale:.2f}")
    print(f"| Max Safety Orders: {safety_order_max_count}")
    print(f"| Initial Price Deviation (SO1): {price_deviation_initial * 100:.2f}%")
    print(f"| Safety Order Step Scale: {safety_order_step_scale:.2f}")
    print(f"| Take Profit Target: {take_profit * 100:.2f}%")
    print("\n--- Summary of Required Funds ---")
    print(f"| Max Stake per Trade (Full SOs): {total_theoretical_stake:.2f} USDT")
    print(f"| Max Price Deviation Tolerance: {max_cumulative_dev * 100:.2f}%")
    print(f"| Tradable Balance Ratio: {tradable_balance_ratio:.2f}")
    print(f"| 💡 REAL WALLET BALANCE REQUIRED: {real_wallet_required:.2f} USDT (per single trade)")
    print("\n📜 Detailed Order Levels:")
    print("-" * 85)
    print(f" {'Order':<4} | {'Price Deviation %':<20} | {'Order Volume (USDT)':<20} | {'Cumulative Volume (USDT)':<25}")
    print("-" * 85)

    for res in detailed_results:
        print(
            f" {res['Order']:<4} | "
            f"{res['Deviation'] * 100:>8.2f}%{'':<11} | "
            f"{res['Amount']:>15.2f}{'':<5} | "
            f"{res['Cumulative_Amount']:>22.2f}"
        )

    print("-" * 85)

    return real_wallet_required

input_config = {
    "strategy_name": "AdvancedDCA",
    "params": {
        "roi": {"0": 100.0},
        "stoploss": {"stoploss": -0.99},
        "trailing": {
            "trailing_stop": False,
            "trailing_stop_positive": None,
            "trailing_stop_positive_offset": 0.0,
            "trailing_only_offset_is_reached": False,
        },
        "max_open_trades": {"max_open_trades": 1},
        "buy": {
            "price_deviation_initial": 0.023,
            "rsi_period": 14,
            "rsi_threshold": 30,
            "safety_order_ratio": 2,
            "safety_order_step_scale": 2.3,
            "safety_order_volume_scale": 2,
            "take_profit": 0.03,
            "base_order_size": 100,
            "safety_order_max_count": 3,
        },
        "tradable_balance_ratio": {
            "tradable_balance_ratio": 0.99
        }
    }
}

required_wallet_balance = calculate_max_dca_stake_final(input_config)
