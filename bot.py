import time
import numpy as np
import pandas as pd
from binance import Binance_API
from indicators import calculate_rsi
from model import train_price_model
from scheduler import sleep_to_next_interval
from trade_utils import set_trade_limits

class TradingBot:
    def __init__(self, api_key, secret_key, symbol='BTCUSDT', interval='1m', history_limit=120):
        self.client = Binance_API(api_key=api_key, secret_key=secret_key, futures=True)
        self.symbol = symbol
        self.interval = interval
        self.history_limit = history_limit
        self.klines = []
        self.last_timestamp = 0
        self.open_position = False
        self.position_side = None
        self.entry_price = 0.0
        self.take_profit = 0.0
        self.stop_loss = 0.0
        self._load_initial_klines()

    def _load_initial_klines(self):
        data = self.client.get_klines(symbol=self.symbol, interval=self.interval, limit=self.history_limit).json()
        if data:
            data.pop()  # remove unfinished current bar
        self.klines = data or []
        self.last_timestamp = self.klines[-1][0] if self.klines else 0

    def _fetch_prices(self):
        recent = self.client.get_klines(symbol=self.symbol, interval=self.interval, limit=2).json()
        if not recent:
            return None, None
        last_closed, current_bar = recent[0], recent[1]

        if last_closed[0] > self.last_timestamp:
            self.klines.append(last_closed)
            self.last_timestamp = last_closed[0]
        if len(self.klines) > self.history_limit:
            self.klines = self.klines[-self.history_limit:]

        closed = np.array([float(k[4]) for k in self.klines])
        live = float(current_bar[4])
        return closed, live

    def run(self):
        while True:
            try:
                sleep_to_next_interval()

                closed, live_price = self._fetch_prices()
                if closed is None or len(closed) < 60:
                    print("Недостаточно данных для обучения модели")
                    continue

                # Вычисление признаков: RSI, Short MA, Long MA
                rsi_series = calculate_rsi(closed)
                short_ma = pd.Series(closed).rolling(window=50, min_periods=1).mean().values
                long_ma = pd.Series(closed).rolling(window=200, min_periods=1).mean().values

                # Формируем обучающее окно из последних 60 точек
                X_train = np.column_stack((
                    rsi_series[-60:].values,
                    short_ma[-60:],
                    long_ma[-60:]
                ))
                y_train = closed[-60:]

                # Обучаем модель и прогнозируем
                model = train_price_model(X_train, y_train)

                # Готовим признаки для следующего прогноза с учётом live price
                extended = np.append(closed, live_price)
                rsi_live = calculate_rsi(extended).iloc[-1]
                ma_series = pd.Series(extended)
                short_live = ma_series.rolling(window=50, min_periods=1).mean().iloc[-1]
                long_live = ma_series.rolling(window=200, min_periods=1).mean().iloc[-1]

                next_price_prediction = model.predict([[rsi_live, short_live, long_live]])[0]
                print('Predicted next price:', next_price_prediction)

                # Текущие индикаторы
                print(f'last_rsi: {rsi_live:.2f}')
                if self.open_position:
                    # P/L
                    profit = (live_price - self.entry_price) if self.position_side == 'BUY' else (self.entry_price - live_price)
                    profit_pct = profit / self.entry_price * 100
                    print(f'Position P/L: {profit_pct:.2f}%')

                # Логика работы с позициями
                self._update_position(live_price, rsi_live, next_price_prediction)

            except Exception as exc:
                print(f"Ошибка: {exc}")
                time.sleep(10)

    def _update_position(self, price, rsi, prediction):
        if self.open_position:
            if self.position_side == 'BUY' and (price >= self.take_profit or price <= self.stop_loss):
                print('CLOSE LONG POSITION!')
                self.client.close_position(symbol=self.symbol, side='SELL', qnt=1)
                self.open_position = False
                self.position_side = None
            elif self.position_side == 'SELL' and (price <= self.take_profit or price >= self.stop_loss):
                print('CLOSE SHORT POSITION!')
                self.client.close_position(symbol=self.symbol, side='BUY', qnt=1)
                self.open_position = False
                self.position_side = None
        else:
            tp, sl = set_trade_limits(price, 0.01, 0.005)
            if rsi <= 30 and prediction > price:
                print('ENTER LONG!')
                self.client.post_market_order(symbol=self.symbol, side='BUY', qnt=1)
                self.open_position, self.position_side = True, 'BUY'
                self.entry_price, self.take_profit, self.stop_loss = price, tp, sl
            elif rsi >= 70 and prediction < price:
                print('ENTER SHORT!')
                self.client.post_market_order(symbol=self.symbol, side='SELL', qnt=1)
                self.open_position, self.position_side = True, 'SELL'
                self.entry_price, self.take_profit, self.stop_loss = price, tp, sl
            else:
                print('NO SIGNAL')