def set_trade_limits(entry_price, tp_ratio, sl_ratio):
    return entry_price * (1 + tp_ratio), entry_price * (1 - sl_ratio)