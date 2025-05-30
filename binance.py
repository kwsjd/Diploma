import hashlib
import hmac
import time
import urllib.parse
from requests import request

class Binance_API:
    spot_link = 'https://api.binance.com'
    futures_link = 'https://fapi.binance.com'

    def __init__(self, api_key="test", secret_key="test", futures=False):
        self.api_key = api_key
        self.secret_key = secret_key
        self.futures = futures
        self.base_link = self.futures_link if self.futures else self.spot_link
        self.header = {'X-MBX-APIKEY': self.api_key}

    def gen_signature(self, params):
        params_str = urllib.parse.urlencode(params)
        sign = hmac.new(bytes(self.secret_key, "utf-8"), params_str.encode("utf-8"), hashlib.sha256).hexdigest()
        return sign

    def http_request(self, method, endpoint, params=None, sign_need=False):
        if params is None:
            params = {}
        # Добавляем в словарь, если необходимо, параметры для подписи - отпечаток времени и саму подпись.
        if sign_need:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self.gen_signature(params)

        url = self.base_link + endpoint
        response = request(method=method, url=url, params=params, headers=self.header)
        return response
        if response:  # Проверяем если ответ не пустой - чтоб не получить ошибки форматирования пустого ответа.
            response = response.json()
        else:
            print(response.text)
        return response

    def get_open_interest_futures(self, symbol):
        method = 'GET'
        endpoint = '/fapi/v1/openInterest'
        params = {'symbol': symbol}
        return self.http_request(endpoint=endpoint, method=method, params=params)

    def get_klines(self, symbol: str, interval: str, startTime=None, endTime=None, limit=500):
        method = 'GET'
        endpoint = '/fapi/v1/klines' if self.futures else '/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        if startTime:
            params['startTime'] = startTime
        if endTime:
            params['endTime'] = endTime
        return self.http_request(endpoint=endpoint, method=method, params=params)

    def get_exchange_info(self, symbol: str = None, symbols=None):
        if self.futures:
            endpoint = '/fapi/v1/exchangeInfo'
        else:
            endpoint = '/api/v3/exchangeInfo'
        method = "GET"
        params = {}
        if symbol and not symbols:
            params['symbol'] = symbol
            return self.http_request(endpoint=endpoint, method=method, params=params)

        if symbols and not symbol:
            params['symbols'] = symbols
            return self.http_request(endpoint=endpoint, method=method, params=params)

        return self.http_request(endpoint=endpoint, method=method, params=params)

    def post_limit_order(self, symbol, side, quantity, price, time_in_force="GTC", reduceOnly=False):
        endpoint = '/fapi/v1/order' if self.futures else '/api/v3/order'
        method = "POST"
        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "quantity": quantity,
            "price": price,
            "timeInForce": time_in_force
        }
        if reduceOnly:
            params['reduceOnly'] = reduceOnly
        return self.http_request(endpoint=endpoint, method=method, params=params, sign_need=True)

    def post_market_order(self, symbol: str, side, qnt: float = None, quoteOrderQty: float = None):
        if not qnt and not quoteOrderQty:
            raise Exception("Не указан один из обязательных параметров quantity или quoteOrderQty")

        if self.futures:
            endpoint = '/fapi/v1/order'
        else:
            endpoint = '/api/v3/order'

        method = 'POST'
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
        }
        if qnt or quoteOrderQty:
            if qnt:
                params['quantity'] = qnt
            if quoteOrderQty:
                params['quoteOrderQty'] = quoteOrderQty

        return self.http_request(endpoint=endpoint, method=method, params=params, sign_need=True)

    def delete_cancel_order(self, symbol: str, orderId: int = None, origClientOrderId: str = None):
        if self.futures:
            endpoint = "/fapi/v1/order"
        else:
            endpoint = "/api/v3/order"

        method = 'DELETE'
        params = {
            'symbol': symbol,
        }

        if orderId or origClientOrderId:
            if orderId:
                params['orderId'] = orderId
            if origClientOrderId:
                params['origClientOrderId'] = origClientOrderId
        else:
            print("Не указан один из обязательных параметров orderId или origClientOrderId")
            return

        return self.http_request(endpoint=endpoint, method=method, params=params, sign_need=True)

    def delete_cancel_all_open_orders(self, symbol: str):
        if self.futures:
            endpoint = "/fapi/v1/allOpenOrders"
        else:
            endpoint = "/api/v3/openOrders"

        method = 'DELETE'
        params = {
            'symbol': symbol,
        }

        return self.http_request(endpoint=endpoint, method=method, params=params, sign_need=True)

    def get_acc_info_userdata(self):
        if self.futures:
            endpoint = "/fapi/v2/account"
        else:
            endpoint = "/api/v3/account"

        method = 'GET'

        return self.http_request(endpoint=endpoint, method=method, sign_need=True)

    def close_position(self, symbol: str, side: str, qnt: float = None):
        """
        Закрытие позиции по рыночной цене.
        """
        opposite_side = 'SELL' if side == 'BUY' else 'BUY'
        return self.post_market_order(symbol=symbol, side=opposite_side, qnt=qnt)
