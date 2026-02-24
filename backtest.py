import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from binance import Binance_API
from indicators import calculate_rsi
from model import train_price_model
from trade_utils import set_trade_limits


INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


@dataclass
class Trade:
    side: str
    entry_time: int
    entry_price: float
    exit_time: int
    exit_price: float
    pnl_pct: float


class StrategyBacktester:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        history_limit: int = 120,
        tp_ratio: float = 0.01,
        sl_ratio: float = 0.005,
        fee_pct: float = 0.04,
        futures: bool = True,
    ):
        if interval not in INTERVAL_MS:
            raise ValueError(f"Unsupported interval: {interval}")
        self.client = Binance_API(api_key="test", secret_key="test", futures=futures)
        self.symbol = symbol
        self.interval = interval
        self.history_limit = history_limit
        self.tp_ratio = tp_ratio
        self.sl_ratio = sl_ratio
        self.fee_pct = fee_pct

    def _download_klines(self, start_ms: int, end_ms: int) -> List[list]:
        step = INTERVAL_MS[self.interval]
        cursor = start_ms
        all_rows: List[list] = []

        while cursor < end_ms:
            response = self.client.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                startTime=cursor,
                endTime=end_ms,
                limit=1000,
            )
            rows = response.json()
            if not rows:
                break

            all_rows.extend(rows)
            last_open = rows[-1][0]
            cursor = last_open + step

            if len(rows) < 1000:
                break

        dedup = {row[0]: row for row in all_rows}
        ordered = [dedup[k] for k in sorted(dedup.keys())]
        return ordered

    def run(self, days: int) -> dict:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        klines = self._download_klines(start_ms=start_ms, end_ms=end_ms)
        if len(klines) < self.history_limit + 60:
            raise RuntimeError("Недостаточно свечей для бэктеста.")

        close_prices = np.array([float(k[4]) for k in klines], dtype=float)
        open_times = np.array([int(k[0]) for k in klines], dtype=int)

        open_position = False
        position_side: Optional[str] = None
        entry_price = 0.0
        entry_time = 0
        take_profit = 0.0
        stop_loss = 0.0
        trades: List[Trade] = []
        equity = 1.0

        for i in range(self.history_limit, len(close_prices)):
            history = close_prices[max(0, i - self.history_limit):i]
            price = close_prices[i]
            ts = open_times[i]

            if len(history) < 60:
                continue

            rsi_series = calculate_rsi(history)
            short_ma = pd.Series(history).rolling(window=50, min_periods=1).mean().values
            long_ma = pd.Series(history).rolling(window=200, min_periods=1).mean().values

            X_train = np.column_stack((
                rsi_series[-60:].values,
                short_ma[-60:],
                long_ma[-60:],
            ))
            y_train = history[-60:]
            model = train_price_model(X_train, y_train)

            extended = np.append(history, price)
            rsi_live = calculate_rsi(extended).iloc[-1]
            ma_series = pd.Series(extended)
            short_live = ma_series.rolling(window=50, min_periods=1).mean().iloc[-1]
            long_live = ma_series.rolling(window=200, min_periods=1).mean().iloc[-1]
            prediction = model.predict([[rsi_live, short_live, long_live]])[0]

            if open_position:
                should_close = False
                if position_side == "BUY" and (price >= take_profit or price <= stop_loss):
                    should_close = True
                elif position_side == "SELL" and (price <= take_profit or price >= stop_loss):
                    should_close = True

                if should_close:
                    raw_pnl_pct = (
                        (price - entry_price) / entry_price * 100
                        if position_side == "BUY"
                        else (entry_price - price) / entry_price * 100
                    )
                    net_pnl_pct = raw_pnl_pct - (2 * self.fee_pct)
                    equity *= (1 + net_pnl_pct / 100)
                    trades.append(
                        Trade(
                            side=position_side,
                            entry_time=entry_time,
                            entry_price=entry_price,
                            exit_time=ts,
                            exit_price=price,
                            pnl_pct=net_pnl_pct,
                        )
                    )
                    open_position = False
                    position_side = None
                continue

            tp, sl = set_trade_limits(price, self.tp_ratio, self.sl_ratio)
            if rsi_live <= 30 and prediction > price:
                open_position = True
                position_side = "BUY"
                entry_price = price
                entry_time = ts
                take_profit = tp
                stop_loss = sl
            elif rsi_live >= 70 and prediction < price:
                open_position = True
                position_side = "SELL"
                entry_price = price
                entry_time = ts
                take_profit = tp
                stop_loss = sl

        if open_position:
            final_price = close_prices[-1]
            final_ts = open_times[-1]
            raw_pnl_pct = (
                (final_price - entry_price) / entry_price * 100
                if position_side == "BUY"
                else (entry_price - final_price) / entry_price * 100
            )
            net_pnl_pct = raw_pnl_pct - (2 * self.fee_pct)
            equity *= (1 + net_pnl_pct / 100)
            trades.append(
                Trade(
                    side=position_side or "UNKNOWN",
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=final_ts,
                    exit_price=final_price,
                    pnl_pct=net_pnl_pct,
                )
            )

        total_return_pct = (equity - 1) * 100
        win_rate = (sum(1 for t in trades if t.pnl_pct > 0) / len(trades) * 100) if trades else 0.0
        avg_trade_pct = (sum(t.pnl_pct for t in trades) / len(trades)) if trades else 0.0

        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "days": days,
            "candles": len(klines),
            "trades": trades,
            "total_return_pct": total_return_pct,
            "win_rate": win_rate,
            "avg_trade_pct": avg_trade_pct,
            "start": start_dt,
            "end": end_dt,
        }


def _fmt_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def main():
    parser = argparse.ArgumentParser(description="Бэктест текущей стратегии на исторических свечах Binance")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--history-limit", type=int, default=120)
    parser.add_argument("--tp", type=float, default=0.01, help="Take-profit в долях (0.01 = 1%)")
    parser.add_argument("--sl", type=float, default=0.005, help="Stop-loss в долях (0.005 = 0.5%)")
    parser.add_argument("--fee", type=float, default=0.04, help="Комиссия на одну сторону в процентах")
    parser.add_argument("--spot", action="store_true", help="Использовать spot klines вместо futures")
    args = parser.parse_args()

    backtester = StrategyBacktester(
        symbol=args.symbol,
        interval=args.interval,
        history_limit=args.history_limit,
        tp_ratio=args.tp,
        sl_ratio=args.sl,
        fee_pct=args.fee,
        futures=not args.spot,
    )
    result = backtester.run(days=args.days)

    print(f"Период: {result['start']} — {result['end']}")
    print(f"Инструмент: {result['symbol']} {result['interval']}; свечей: {result['candles']}")
    print(f"Сделок: {len(result['trades'])}")
    print(f"Доходность: {result['total_return_pct']:.2f}%")
    print(f"WinRate: {result['win_rate']:.2f}%")
    print(f"Средняя сделка: {result['avg_trade_pct']:.4f}%")

    if result["trades"]:
        print("\nПоследние 5 сделок:")
        for tr in result["trades"][-5:]:
            print(
                f"{tr.side:<4} | {_fmt_ms(tr.entry_time)} -> {_fmt_ms(tr.exit_time)} | "
                f"{tr.entry_price:.2f} -> {tr.exit_price:.2f} | {tr.pnl_pct:.3f}%"
            )


if __name__ == "__main__":
    main()
