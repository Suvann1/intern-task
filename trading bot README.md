Simple Binance Testnet Bot
Setup Steps
Open a terminal in the folder containing these files.

Install dependencies by running this command:
pip install -r requirements.txt

Create a file named .env in the exact same folder.

Paste your API keys inside the .env file exactly like this:
BINANCE_API_KEY=your_testnet_api_key_here
BINANCE_API_SECRET=your_testnet_api_secret_here

How to Run Examples
Place a MARKET Order (Buy 0.05 BTC):
python bot.py -s BTCUSDT --side BUY -t MARKET -q 0.05

Place a LIMIT Order (Sell 1.5 ETH at $3500):
python bot.py -s ETHUSDT --side SELL -t LIMIT -q 1.5 -p 3500

Assumptions
You have valid Binance Futures Testnet API credentials.

You have sufficient USDT balance on the Testnet to execute these trades.

You are running Python 3.x and have pip installed.

FILE 2: Name this file exactly requirements.txt and paste everything below:

python-binance==1.0.19
python-dotenv==1.0.1
