# Indian Stock Market Prediction System 📈

A comprehensive Python-based stock market prediction system for Indian stocks (NSE) that uses Machine Learning models to generate buy/sell signals with technical analysis.

## 🎯 Features

- **Dual ML Models**: Linear Regression + Random Forest Regressor for ensemble predictions
- **Technical Indicators**: RSI, MACD, Moving Averages, Bollinger Bands, ROC
- **Buy/Sell Signals**: Intelligent scoring system combining ML predictions and technical analysis
- **Comprehensive Visualizations**: 
  - Model prediction accuracy charts
  - Technical analysis with indicators
  - Price trends and patterns
- **Indian Market Focus**: Pre-configured for NSE stocks (.NS suffix)

## 📊 Key Components

### 1. Machine Learning Models
- **Linear Regression**: Fast baseline model for price prediction
- **Random Forest**: Advanced ensemble model for better accuracy
- **Feature Engineering**: 
  - Lag features (previous 5 days)
  - Technical indicators
  - Volume analysis

### 2. Technical Indicators
- **Moving Averages**: MA 5, 20, 50 for trend identification
- **RSI**: Relative Strength Index for overbought/oversold conditions
- **MACD**: Moving Average Convergence Divergence for momentum
- **Bollinger Bands**: Volatility and price channel analysis
- **ROC**: Rate of Change for momentum measurement

### 3. Signal Generation
The system uses a sophisticated scoring algorithm:

| Signal | Score | Meaning |
|--------|-------|---------|
| STRONG BUY | ≥ 3 | Multiple bullish indicators align |
| BUY | 1-2 | Moderate bullish sentiment |
| HOLD | 0 | Neutral market conditions |
| SELL | -1 to -2 | Moderate bearish sentiment |
| STRONG SELL | ≤ -3 | Multiple bearish indicators align |

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from stock_predictor import IndianStockPredictor

# Initialize predictor
predictor = IndianStockPredictor(ticker="RELIANCE.NS", period='2y')

# Fetch and analyze data
predictor.fetch_data()
predictor.calculate_technical_indicators()

# Train ML models
predictor.train_models()

# Generate trading signals
signal = predictor.generate_signals()

# Create visualizations
predictor.plot_predictions()
predictor.plot_technical_analysis()
```

## 📈 Popular Indian Stocks (NSE)

Use these ticker symbols:

| Symbol | Company |
|--------|---------|
| RELIANCE.NS | Reliance Industries |
| TCS.NS | Tata Consultancy Services |
| INFY.NS | Infosys |
| HDFCBANK.NS | HDFC Bank |
| ICICIBANK.NS | ICICI Bank |
| HINDUNILVR.NS | Hindustan Unilever |
| SBIN.NS | State Bank of India |
| BHARTIARTL.NS | Bharti Airtel |
| ITC.NS | ITC Limited |
| KOTAKBANK.NS | Kotak Mahindra Bank |

## 💡 Example Workflow

```python
# Analyze multiple stocks
stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']

for stock in stocks:
    print(f"\n{'='*60}")
    print(f"Analyzing {stock}")
    print(f"{'='*60}")
    
    predictor = IndianStockPredictor(stock, period='1y')
    predictor.fetch_data()
    predictor.calculate_technical_indicators()
    predictor.train_models()
    
    signal_data = predictor.generate_signals()
    
    # Save charts
    predictor.plot_predictions(f'predictions_{stock}.png')
    predictor.plot_technical_analysis(f'technical_{stock}.png')
```

## 📊 Model Evaluation Metrics

The system evaluates models using:
- **MAE (Mean Absolute Error)**: Average prediction error in ₹
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors
- **R² Score**: Model fit quality (0-1, higher is better)

## 🎨 Visualizations Generated

### 1. Predictions Chart
- Linear Regression predictions vs actual
- Random Forest predictions vs actual
- Error distribution analysis
- Scatter plot: predicted vs actual

### 2. Technical Analysis Chart
- Price with Moving Averages & Bollinger Bands
- RSI with overbought/oversold zones
- MACD with signal line and histogram
- Trading volume analysis

## ⚙️ Customization

### Change Time Period
```python
# 1 year
predictor = IndianStockPredictor("RELIANCE.NS", period='1y')

# 5 years
predictor = IndianStockPredictor("RELIANCE.NS", period='5y')

# Maximum available data
predictor = IndianStockPredictor("RELIANCE.NS", period='max')
```

### Adjust Signal Sensitivity
Modify the scoring thresholds in the `generate_signals()` method:
```python
# More conservative (harder to trigger buy/sell)
if signal_score >= 4:  # Instead of 3
    signal = "STRONG BUY"
```

## 📝 Skills Demonstrated

✅ **Data Preprocessing**: Handling time series data, feature engineering  
✅ **Machine Learning**: Regression models, ensemble methods  
✅ **Model Evaluation**: MAE, RMSE, R² metrics  
✅ **Technical Analysis**: Implementation of trading indicators  
✅ **Data Visualization**: Matplotlib, Seaborn for insights  
✅ **Python Libraries**: pandas, numpy, scikit-learn, yfinance  
✅ **Financial Domain**: Understanding of market dynamics  

## ⚠️ Disclaimer

**This is an educational project for learning purposes only.**

- Not financial advice
- Past performance doesn't guarantee future results
- Always do your own research (DYOR)
- Consult with financial advisors before trading
- Use at your own risk

## 🔧 Technical Architecture

```
stock_predictor.py
├── IndianStockPredictor (Main Class)
│   ├── fetch_data()              # Get historical data
│   ├── calculate_technical_indicators()  # Compute indicators
│   ├── prepare_features()        # Feature engineering
│   ├── train_models()            # ML model training
│   ├── generate_signals()        # Buy/sell logic
│   ├── plot_predictions()        # Prediction charts
│   └── plot_technical_analysis() # Technical charts
└── main()                        # Entry point
```

## 📚 Next Steps

1. **Backtesting**: Implement historical strategy testing
2. **Real-time Data**: Add live market data integration
3. **Portfolio Management**: Multi-stock portfolio optimization
4. **Alerts**: Email/SMS notifications for signals
5. **Web Dashboard**: Interactive Streamlit/Dash interface
6. **Advanced Models**: LSTM, Prophet for time series
7. **Risk Management**: Stop-loss, position sizing

## 🤝 Contributing

Suggestions for improvement:
- Add more technical indicators (Stochastic, ADX, etc.)
- Implement sentiment analysis from news
- Add fundamental analysis metrics
- Create mobile app version
- Build API for integration

## 📞 Support

For questions or issues:
- Review the code comments
- Check Yahoo Finance API documentation
- Verify stock ticker symbols at NSE website

---

**Built with Python, Machine Learning & Technical Analysis** 🐍📊

*Remember: The best prediction is one made with caution and proper research!*
