import os
import sys
import logging
import argparse
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

def setup_logger():
    logger = logging.getLogger("TradingBot")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("bot.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) 
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()

def validate_inputs(symbol: str, side: str, order_type: str, quantity: float, price: float = None):
    if side.upper() not in ['BUY', 'SELL']:
        raise ValueError(f"Invalid side: '{side}'. Must be BUY or SELL.")
    if order_type.upper() not in ['MARKET', 'LIMIT']:
        raise ValueError(f"Invalid order type: '{order_type}'. Must be MARKET or LIMIT.")
    if quantity <= 0:
        raise ValueError("Quantity must be strictly greater than 0.")
    if order_type.upper() == 'LIMIT' and (price is None or price <= 0):
        raise ValueError("LIMIT orders require a valid price greater than 0.")
    return True

def place_order(symbol: str, side: str, order_type: str, quantity: float, price: float = None):
    try:
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')

        if not api_key or not api_secret:
            raise ValueError("API credentials missing. Please set BINANCE_API_KEY and BINANCE_API_SECRET in your .env file.")

        validate_inputs(symbol, side, order_type, quantity, price)

        client = Client(api_key, api_secret, testnet=True)

        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': quantity
        }
        
        if order_type.upper() == 'LIMIT':
            params['price'] = price
            params['timeInForce'] = 'GTC'

        response = client.futures_create_order(**params)

        logger.info("\n[SUCCESS] Order placed successfully!")
        logger.info(f" -> Order ID:     {response.get('orderId')}")
        logger.info(f" -> Status:       {response.get('status')}")
        logger.info(f" -> Executed Qty: {response.get('executedQty')}")
        
        avg_price = response.get('avgPrice')
        if avg_price and float(avg_price) > 0:
            logger.info(f" -> Avg Price:    {avg_price}")

    except BinanceAPIException as e:
        logger.error(f"\n[FAILED] API Error: {e.message} (Code: {e.status_code})")
    except BinanceRequestException as e:
        logger.error("\n[FAILED] Network Error. Check your connection.")
    except ValueError as e:
        logger.error(f"\n[FAILED] Invalid Input: {e}")
    except Exception as e:
        logger.error(f"\n[FAILED] An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-File Binance Futures Testnet Trading Bot",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Examples:\n  Market Order: python bot.py -s BTCUSDT --side BUY -t MARKET -q 0.1\n  Limit Order:  python bot.py -s ETHUSDT --side SELL -t LIMIT -q 1.5 -p 3500"
    )

    parser.add_argument('-s', '--symbol', type=str, required=True)
    parser.add_argument('--side', type=str, required=True, choices=['BUY', 'SELL', 'buy', 'sell'])
    parser.add_argument('-t', '--type', type=str, required=True, choices=['MARKET', 'LIMIT', 'market', 'limit'], dest="order_type")
    parser.add_argument('-q', '--quantity', type=float, required=True)
    parser.add_argument('-p', '--price', type=float, required=False)

    args = parser.parse_args()

    print("      BINANCE FUTURES TESTNET BOT       ")
    
    
    place_order(
        symbol=args.symbol,
        side=args.side,
        order_type=args.order_type,
        quantity=args.quantity,
        price=args.price
    )
    
    print("========================================\n")
