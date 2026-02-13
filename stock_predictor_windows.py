import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class IndianStockPredictor:
    """Stock Market Prediction System for Indian Market"""
    
    def __init__(self, ticker, period='2y'):
        """
        Initialize the predictor
        ticker: Stock symbol with .NS suffix (e.g., 'RELIANCE.NS', 'TCS.NS')
        period: Data period ('1y', '2y', '5y', 'max')
        """
        self.ticker = ticker
        self.period = period
        self.data = None
        self.model_lr = None
        self.model_rf = None
        self.predictions = None
        
    def fetch_data(self):
        """Fetch historical stock data from Yahoo Finance with robust error handling"""
        print(f"Fetching data for {self.ticker}...")
        
        import time
        
        # List of tickers to try (original, alternate exchange, and fallbacks)
        tickers_to_try = [self.ticker]
        
        # Add alternate exchange if applicable
        if '.NS' in self.ticker:
            tickers_to_try.append(self.ticker.replace('.NS', '.BO'))
        elif '.BO' in self.ticker:
            tickers_to_try.append(self.ticker.replace('.BO', '.NS'))
        
        # Add some fallback US stocks if all else fails
        if not any(x in self.ticker for x in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']):
            tickers_to_try.extend(['MSFT', 'GOOGL'])
        
        last_error = None
        
        for i, ticker in enumerate(tickers_to_try):
            try:
                if i > 0:
                    print(f"Trying alternate: {ticker}...")
                
                time.sleep(1)  # Rate limiting
                
                stock = yf.Ticker(ticker)
                
                # Try to fetch data
                self.data = stock.history(period=self.period)
                
                # Check if we got data
                if not self.data.empty and len(self.data) > 100:
                    print(f"✓ Success! Using {ticker}")
                    print(f"  Data points: {len(self.data)}")
                    print(f"  Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
                    self.ticker = ticker
                    return self.data
                else:
                    print(f"✗ No sufficient data for {ticker} (got {len(self.data) if self.data is not None else 0} points)")
                    
            except Exception as e:
                last_error = str(e)
                print(f"✗ Error with {ticker}: {last_error}")
                continue
        
        # If we get here, nothing worked
        raise ValueError(f"""
Failed to fetch data for any ticker.
Tried: {', '.join(tickers_to_try)}
Last error: {last_error}

Troubleshooting:
1. Check your internet connection
2. Try running diagnostic_test.py first
3. Make sure yfinance is updated: pip install --upgrade yfinance
4. Try again in a few minutes (API might be rate-limited)
""")
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators for better predictions"""
        df = self.data.copy()
        
        # Moving Averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Price Rate of Change
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        self.data = df
        print("✓ Technical indicators calculated")
        return df
    
    def prepare_features(self):
        """Prepare features for ML models"""
        df = self.data.copy()
        
        # Create lag features
        for i in range(1, 6):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
            df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
        
        # Target: Next day's closing price
        df['Target'] = df['Close'].shift(-1)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Select features
        feature_cols = ['Close', 'Open', 'High', 'Low', 'Volume',
                       'MA_5', 'MA_20', 'MA_50', 'RSI', 'MACD',
                       'BB_Upper', 'BB_Lower', 'ROC', 'Volume_MA'] + \
                       [f'Close_Lag_{i}' for i in range(1, 6)]
        
        X = df[feature_cols]
        y = df['Target']
        
        return X, y, df
    
    def train_models(self):
        """Train multiple regression models"""
        print("\nTraining models...")
        
        X, y, df = self.prepare_features()
        
        print(f"  Features shape: {X.shape}")
        print(f"  Training samples: {len(X)}")
        
        # Split data (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Linear Regression
        print("  Training Linear Regression...")
        self.model_lr = LinearRegression()
        self.model_lr.fit(X_train, y_train)
        
        # Random Forest Regressor
        print("  Training Random Forest...")
        self.model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model_rf.fit(X_train, y_train)
        
        # Predictions
        lr_pred = self.model_lr.predict(X_test)
        rf_pred = self.model_rf.predict(X_test)
        
        # Store results
        self.predictions = {
            'X_test': X_test,
            'y_test': y_test,
            'lr_pred': lr_pred,
            'rf_pred': rf_pred,
            'dates': df.index[-len(y_test):]
        }
        
        # Evaluate models
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        models = {
            'Linear Regression': lr_pred,
            'Random Forest': rf_pred
        }
        
        for name, pred in models.items():
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)
            
            print(f"\n{name}:")
            print(f"  MAE:  ${mae:.2f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  R²:   {r2:.4f}")
        
        return self.predictions
    
    def generate_signals(self, lookback=30):
        """Generate buy/sell signals based on predictions and technical indicators"""
        df = self.data.copy()
        
        # Get latest data for prediction
        X, _, _ = self.prepare_features()
        latest_features = X.iloc[-1:].values
        
        # Predict next day price
        predicted_price_lr = self.model_lr.predict(latest_features)[0]
        predicted_price_rf = self.model_rf.predict(latest_features)[0]
        predicted_price = (predicted_price_lr + predicted_price_rf) / 2  # Ensemble
        
        current_price = df['Close'].iloc[-1]
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        # Get technical indicators for latest day
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        signal_line = df['Signal_Line'].iloc[-1]
        ma_20 = df['MA_20'].iloc[-1]
        ma_50 = df['MA_50'].iloc[-1]
        
        # Signal scoring system
        signal_score = 0
        reasons = []
        
        # Price prediction signal
        if price_change > 2:
            signal_score += 2
            reasons.append(f"Strong upward prediction (+{price_change:.2f}%)")
        elif price_change > 0.5:
            signal_score += 1
            reasons.append(f"Moderate upward prediction (+{price_change:.2f}%)")
        elif price_change < -2:
            signal_score -= 2
            reasons.append(f"Strong downward prediction ({price_change:.2f}%)")
        elif price_change < -0.5:
            signal_score -= 1
            reasons.append(f"Moderate downward prediction ({price_change:.2f}%)")
        
        # RSI signals
        if rsi < 30:
            signal_score += 2
            reasons.append(f"Oversold (RSI: {rsi:.1f})")
        elif rsi < 40:
            signal_score += 1
            reasons.append(f"Near oversold (RSI: {rsi:.1f})")
        elif rsi > 70:
            signal_score -= 2
            reasons.append(f"Overbought (RSI: {rsi:.1f})")
        elif rsi > 60:
            signal_score -= 1
            reasons.append(f"Near overbought (RSI: {rsi:.1f})")
        
        # MACD signals
        if macd > signal_line and macd > 0:
            signal_score += 1
            reasons.append("Bullish MACD crossover")
        elif macd < signal_line and macd < 0:
            signal_score -= 1
            reasons.append("Bearish MACD crossover")
        
        # Moving average signals
        if current_price > ma_20 > ma_50:
            signal_score += 1
            reasons.append("Price above MA20 and MA50")
        elif current_price < ma_20 < ma_50:
            signal_score -= 1
            reasons.append("Price below MA20 and MA50")
        
        # Final signal
        if signal_score >= 3:
            signal = "STRONG BUY"
            color = "🟢"
        elif signal_score >= 1:
            signal = "BUY"
            color = "🟩"
        elif signal_score <= -3:
            signal = "STRONG SELL"
            color = "🔴"
        elif signal_score <= -1:
            signal = "SELL"
            color = "🟥"
        else:
            signal = "HOLD"
            color = "🟡"
        
        # Print signal report
        print("\n" + "="*60)
        print("TRADING SIGNAL REPORT")
        print("="*60)
        print(f"Stock: {self.ticker}")
        print(f"Date: {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Price: ${predicted_price:.2f} ({price_change:+.2f}%)")
        print(f"\n{color} SIGNAL: {signal} (Score: {signal_score})")
        print("\nReasons:")
        for i, reason in enumerate(reasons, 1):
            print(f"  {i}. {reason}")
        print("="*60)
        
        return {
            'signal': signal,
            'score': signal_score,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change_pct': price_change,
            'reasons': reasons,
            'rsi': rsi,
            'macd': macd
        }
    
    def plot_predictions(self, save_path='outputs/predictions.png'):
        """Visualize predictions vs actual prices"""
        if self.predictions is None:
            raise ValueError("Train models first!")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Stock Price Predictions - {self.ticker}', fontsize=16, fontweight='bold')
        
        dates = self.predictions['dates']
        y_test = self.predictions['y_test']
        lr_pred = self.predictions['lr_pred']
        rf_pred = self.predictions['rf_pred']
        
        # Plot 1: Linear Regression Predictions
        axes[0, 0].plot(dates, y_test, label='Actual Price', color='blue', linewidth=2)
        axes[0, 0].plot(dates, lr_pred, label='Predicted Price', color='red', linewidth=2, linestyle='--')
        axes[0, 0].set_title('Linear Regression Predictions')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Random Forest Predictions
        axes[0, 1].plot(dates, y_test, label='Actual Price', color='blue', linewidth=2)
        axes[0, 1].plot(dates, rf_pred, label='Predicted Price', color='green', linewidth=2, linestyle='--')
        axes[0, 1].set_title('Random Forest Predictions')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Prediction Error Distribution
        lr_errors = y_test - lr_pred
        rf_errors = y_test - rf_pred
        
        axes[1, 0].hist(lr_errors, bins=30, alpha=0.6, label='Linear Regression', color='red')
        axes[1, 0].hist(rf_errors, bins=30, alpha=0.6, label='Random Forest', color='green')
        axes[1, 0].set_title('Prediction Error Distribution')
        axes[1, 0].set_xlabel('Error ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        # Plot 4: Actual vs Predicted Scatter
        axes[1, 1].scatter(y_test, lr_pred, alpha=0.5, label='Linear Reg', color='red', s=30)
        axes[1, 1].scatter(y_test, rf_pred, alpha=0.5, label='Random Forest', color='green', s=30)
        axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                       'k--', lw=2, label='Perfect Prediction')
        axes[1, 1].set_title('Actual vs Predicted Prices')
        axes[1, 1].set_xlabel('Actual Price ($)')
        axes[1, 1].set_ylabel('Predicted Price ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Prediction plot saved to: {save_path}")
        plt.close()
    
    def plot_technical_analysis(self, save_path='outputs/technical_analysis.png'):
        """Plot technical indicators"""
        df = self.data.tail(100).copy()  # Last 100 days
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        fig.suptitle(f'Technical Analysis - {self.ticker}', fontsize=16, fontweight='bold')
        
        # Plot 1: Price with Moving Averages
        axes[0].plot(df.index, df['Close'], label='Close Price', color='black', linewidth=2)
        axes[0].plot(df.index, df['MA_5'], label='MA 5', color='blue', alpha=0.7)
        axes[0].plot(df.index, df['MA_20'], label='MA 20', color='orange', alpha=0.7)
        axes[0].plot(df.index, df['MA_50'], label='MA 50', color='red', alpha=0.7)
        axes[0].fill_between(df.index, df['BB_Lower'], df['BB_Upper'], alpha=0.2, color='gray', label='Bollinger Bands')
        axes[0].set_title('Price with Moving Averages & Bollinger Bands')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: RSI
        axes[1].plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=2)
        axes[1].axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
        axes[1].axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
        axes[1].fill_between(df.index, 30, 70, alpha=0.1, color='gray')
        axes[1].set_title('Relative Strength Index (RSI)')
        axes[1].set_ylabel('RSI')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)
        
        # Plot 3: MACD
        axes[2].plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=2)
        axes[2].plot(df.index, df['Signal_Line'], label='Signal Line', color='red', linewidth=2)
        axes[2].bar(df.index, df['MACD'] - df['Signal_Line'], label='Histogram', 
                   color=['green' if x > 0 else 'red' for x in (df['MACD'] - df['Signal_Line'])], alpha=0.3)
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[2].set_title('MACD (Moving Average Convergence Divergence)')
        axes[2].set_ylabel('MACD')
        axes[2].legend(loc='upper left')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Volume
        colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
                 for i in range(len(df))]
        axes[3].bar(df.index, df['Volume'], color=colors, alpha=0.6)
        axes[3].plot(df.index, df['Volume_MA'], label='Volume MA', color='blue', linewidth=2)
        axes[3].set_title('Trading Volume')
        axes[3].set_xlabel('Date')
        axes[3].set_ylabel('Volume')
        axes[3].legend(loc='upper left')
        axes[3].grid(True, alpha=0.3)
        
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Technical analysis plot saved to: {save_path}")
        plt.close()


def main():
    """Main execution function"""
    
    # Create outputs directory if it doesn't exist
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("Created 'outputs' directory")
    
    print("="*60)
    print("STOCK MARKET PREDICTION SYSTEM")
    print("="*60)
    
    # List of tickers to try (in order of preference)
    tickers_to_try = [
        ("TCS.NS", "Tata Consultancy Services (India)"),
        ("INFY.NS", "Infosys (India)"),
        ("RELIANCE.NS", "Reliance Industries (India)"),
        ("MSFT", "Microsoft (US)"),
        ("GOOGL", "Google (US)"),
    ]
    
    print("\nAttempting to analyze stocks...")
    print("(Will try multiple tickers until one works)")
    print("-"*60)
    
    predictor = None
    signal_data = None
    
    for ticker, name in tickers_to_try:
        try:
            print(f"\n{'='*60}")
            print(f"Trying: {ticker} - {name}")
            print('='*60)
            
            # Initialize predictor
            predictor = IndianStockPredictor(ticker=ticker, period="2y")
            
            # Fetch and prepare data
            predictor.fetch_data()
            predictor.calculate_technical_indicators()
            
            # Train models
            predictor.train_models()
            
            # Generate trading signals
            signal_data = predictor.generate_signals()
            
            # Create visualizations
            predictor.plot_predictions()
            predictor.plot_technical_analysis()
            
            print("\n" + "="*60)
            print("✅ ANALYSIS COMPLETE!")
            print("="*60)
            print(f"\nSuccessfully analyzed: {ticker}")
            print("\nGenerated files:")
            print("  1. predictions.png - Model predictions vs actual prices")
            print("  2. technical_analysis.png - Technical indicators")
            print("\nNext steps:")
            print("  • Review the trading signal above")
            print("  • Check the generated charts")
            print("  • Consider multiple timeframes")
            print("  • Always do your own research!")
            print("\n⚠️  DISCLAIMER: For educational purposes only!")
            
            break  # Success! Stop trying other tickers
            
        except Exception as e:
            print(f"\n✗ Failed with {ticker}: {str(e)}")
            print("Trying next ticker...")
            continue
    
    if predictor is None:
        print("\n" + "="*60)
        print("❌ ALL TICKERS FAILED")
        print("="*60)
        print("\nPlease run diagnostic_test.py to troubleshoot")
        print("\nPossible issues:")
        print("  1. Internet connection problem")
        print("  2. Yahoo Finance API temporarily down")
        print("  3. Firewall blocking requests")
        print("\nTry:")
        print("  • Check internet connection")
        print("  • Run: python diagnostic_test.py")
        print("  • Update yfinance: pip install --upgrade yfinance")
        print("  • Try again in a few minutes")
    
    return predictor, signal_data


if __name__ == "__main__":
    predictor, signal_data = main()
