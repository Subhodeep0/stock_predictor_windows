# Quick Start Guide

Get up and running with the Indian Stock Market Prediction System in minutes!

## 🚀 In 3 Minutes

### Step 1: Install (1 minute)
```bash
pip install -r requirements.txt
```

### Step 2: Run Default Analysis (2 minutes)
```bash
python stock_predictor.py
```

That's it! You'll get:
- ✅ Buy/Sell signal for RELIANCE.NS
- ✅ Price predictions
- ✅ Technical analysis charts
- ✅ Model performance metrics

## 📝 Basic Usage

### Analyze Any Stock

```python
from stock_predictor import IndianStockPredictor

# Create predictor
predictor = IndianStockPredictor("TCS.NS", period='1y')

# Run analysis
predictor.fetch_data()
predictor.calculate_technical_indicators()
predictor.train_models()

# Get signal
signal = predictor.generate_signals()
print(f"Signal: {signal['signal']}")
```

### Popular Stock Tickers

```python
# Banking
"HDFCBANK.NS"   # HDFC Bank
"ICICIBANK.NS"  # ICICI Bank
"SBIN.NS"       # State Bank of India

# IT Services
"TCS.NS"        # Tata Consultancy
"INFY.NS"       # Infosys

# Conglomerate
"RELIANCE.NS"   # Reliance Industries

# FMCG
"HINDUNILVR.NS" # Hindustan Unilever
"ITC.NS"        # ITC Limited

# Telecom
"BHARTIARTL.NS" # Bharti Airtel
```

## 🎯 Common Tasks

### 1. Get Today's Signal

```python
predictor = IndianStockPredictor("INFY.NS")
predictor.fetch_data()
predictor.calculate_technical_indicators()
predictor.train_models()
signal = predictor.generate_signals()

print(f"Signal: {signal['signal']}")
print(f"Current: ₹{signal['current_price']:.2f}")
print(f"Predicted: ₹{signal['predicted_price']:.2f}")
```

### 2. Compare Multiple Stocks

```python
stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']

for ticker in stocks:
    predictor = IndianStockPredictor(ticker)
    predictor.fetch_data()
    predictor.calculate_technical_indicators()
    predictor.train_models()
    signal = predictor.generate_signals()
    print(f"{ticker}: {signal['signal']}")
```

### 3. Analyze Different Time Periods

```python
# Last 6 months
predictor = IndianStockPredictor("SBIN.NS", period='6mo')

# Last 1 year (default)
predictor = IndianStockPredictor("SBIN.NS", period='1y')

# Last 5 years
predictor = IndianStockPredictor("SBIN.NS", period='5y')

# All available data
predictor = IndianStockPredictor("SBIN.NS", period='max')
```

### 4. Generate Charts

```python
predictor = IndianStockPredictor("ITC.NS")
predictor.fetch_data()
predictor.calculate_technical_indicators()
predictor.train_models()

# Create prediction charts
predictor.plot_predictions('my_predictions.png')

# Create technical analysis charts
predictor.plot_technical_analysis('my_technical.png')
```

## 📊 Understanding Signals

| Signal | Meaning | Action |
|--------|---------|--------|
| 🟢 STRONG BUY | Very bullish | Consider buying |
| 🟩 BUY | Moderately bullish | Consider accumulating |
| 🟡 HOLD | Neutral | Wait and watch |
| 🟥 SELL | Moderately bearish | Consider reducing position |
| 🔴 STRONG SELL | Very bearish | Consider exiting |

## 🔍 Reading the Output

### Signal Report Example:
```
============================================================
TRADING SIGNAL REPORT
============================================================
Stock: TCS.NS
Date: 2025-02-10
Current Price: ₹4,234.50
Predicted Price: ₹4,298.75 (+1.52%)

🟩 SIGNAL: BUY (Score: 2)

Reasons:
  1. Moderate upward prediction (+1.52%)
  2. Near oversold (RSI: 42.3)
  3. Price above MA20 and MA50
============================================================
```

### Model Evaluation Example:
```
============================================================
MODEL EVALUATION
============================================================

Linear Regression:
  MAE:  ₹45.23    ← Average error
  RMSE: ₹58.67    ← Penalty for large errors
  R²:   0.9234    ← Fit quality (higher is better)

Random Forest:
  MAE:  ₹38.15    ← Better than Linear Regression
  RMSE: ₹49.32
  R²:   0.9456    ← Excellent fit!
============================================================
```

## ⚡ Quick Tips

1. **Always use .NS suffix** for NSE stocks (e.g., "RELIANCE.NS")
2. **More data = Better predictions** - Use 1-2 years for reliable results
3. **Check both signal and charts** - Visual confirmation is important
4. **Never trade on signals alone** - Always do your own research
5. **Compare timeframes** - Run 6mo, 1y, 2y to see consistency

## 🎓 Next Steps

1. ✅ Run the default analysis
2. ✅ Try different stocks from the list
3. ✅ Check the `example_usage.py` for more examples
4. ✅ Read the full README.md for detailed documentation
5. ✅ Experiment with different time periods

## 🆘 Need Help?

**Can't find data?**
- Check internet connection
- Verify ticker symbol ends with .NS
- Try a different stock

**Errors in code?**
- Make sure all packages are installed: `pip install -r requirements.txt`
- Activate virtual environment if using one

**Want to customize?**
- Check the IndianStockPredictor class in stock_predictor.py
- Modify scoring thresholds in generate_signals() method

## ⚠️ Important Disclaimer

This is an educational tool for learning ML and technical analysis.

- ❌ Not financial advice
- ❌ Not guaranteed to be accurate
- ❌ Past performance ≠ Future results
- ✅ Always do your own research
- ✅ Consult financial advisors

---

**Ready to start? Run this:**

```bash
python stock_predictor.py
```

**Happy analyzing! 📈🚀**
