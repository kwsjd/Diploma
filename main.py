from config import API_KEY, SECRET_KEY
from bot import TradingBot

if __name__ == "__main__":
    bot = TradingBot(api_key=API_KEY, secret_key=SECRET_KEY)
    bot.run()
