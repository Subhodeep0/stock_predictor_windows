"""
Example Usage Script - Indian Stock Market Prediction
Demonstrates various ways to use the stock prediction system
"""

from stock_predictor import IndianStockPredictor

# ============================================================================
# EXAMPLE 1: Single Stock Analysis
# ============================================================================

def example_single_stock():
    """Analyze a single stock with complete workflow"""
    print("EXAMPLE 1: Single Stock Analysis")
    print("="*70)
    
    # Choose a stock
    ticker = "TCS.NS"  # Tata Consultancy Services
    
    # Initialize predictor with 1 year of data
    predictor = IndianStockPredictor(ticker, period='1y')
    
    # Step 1: Fetch historical data
    predictor.fetch_data()
    
    # Step 2: Calculate technical indicators
    predictor.calculate_technical_indicators()
    
    # Step 3: Train ML models
    predictor.train_models()
    
    # Step 4: Generate trading signals
    signal_data = predictor.generate_signals()
    
    # Step 5: Create visualizations
    predictor.plot_predictions(save_path='/mnt/user-data/outputs/tcs_predictions.png')
    predictor.plot_technical_analysis(save_path='/mnt/user-data/outputs/tcs_technical.png')
    
    print(f"\n✅ Analysis complete for {ticker}")
    print(f"Signal: {signal_data['signal']}")
    print(f"Current Price: ₹{signal_data['current_price']:.2f}")
    print(f"Predicted Price: ₹{signal_data['predicted_price']:.2f}")
    
    return predictor, signal_data


# ============================================================================
# EXAMPLE 2: Compare Multiple Stocks
# ============================================================================

def example_multiple_stocks():
    """Compare signals for multiple stocks"""
    print("\n\nEXAMPLE 2: Multiple Stocks Comparison")
    print("="*70)
    
    # List of stocks to analyze
    stocks = [
        'RELIANCE.NS',    # Reliance Industries
        'TCS.NS',         # TCS
        'INFY.NS',        # Infosys
        'HDFCBANK.NS',    # HDFC Bank
    ]
    
    results = {}
    
    for ticker in stocks:
        print(f"\n{'─'*70}")
        print(f"Analyzing: {ticker}")
        print(f"{'─'*70}")
        
        try:
            predictor = IndianStockPredictor(ticker, period='6mo')
            predictor.fetch_data()
            predictor.calculate_technical_indicators()
            predictor.train_models()
            signal_data = predictor.generate_signals()
            
            results[ticker] = {
                'signal': signal_data['signal'],
                'score': signal_data['score'],
                'current_price': signal_data['current_price'],
                'predicted_price': signal_data['predicted_price'],
                'change_pct': signal_data['price_change_pct']
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {e}")
            results[ticker] = {'signal': 'ERROR', 'score': 0}
    
    # Summary Report
    print("\n\n" + "="*70)
    print("SUMMARY REPORT - ALL STOCKS")
    print("="*70)
    print(f"{'Stock':<15} {'Signal':<15} {'Score':<8} {'Current':<12} {'Predicted':<12} {'Change%'}")
    print("─"*70)
    
    for ticker, data in results.items():
        if data['signal'] != 'ERROR':
            print(f"{ticker:<15} {data['signal']:<15} {data['score']:<8} "
                  f"₹{data['current_price']:<11.2f} ₹{data['predicted_price']:<11.2f} "
                  f"{data['change_pct']:+.2f}%")
        else:
            print(f"{ticker:<15} {data['signal']:<15}")
    
    return results


# ============================================================================
# EXAMPLE 3: Deep Analysis with Insights
# ============================================================================

def example_detailed_analysis():
    """Perform detailed analysis with technical insights"""
    print("\n\nEXAMPLE 3: Detailed Technical Analysis")
    print("="*70)
    
    ticker = "INFY.NS"  # Infosys
    
    predictor = IndianStockPredictor(ticker, period='2y')
    predictor.fetch_data()
    predictor.calculate_technical_indicators()
    
    # Access the processed data
    df = predictor.data
    
    # Latest values
    latest = df.iloc[-1]
    
    print(f"\nStock: {ticker}")
    print(f"Date: {df.index[-1].strftime('%Y-%m-%d')}")
    print("\n" + "─"*70)
    print("PRICE INFORMATION")
    print("─"*70)
    print(f"Close Price:     ₹{latest['Close']:.2f}")
    print(f"Open Price:      ₹{latest['Open']:.2f}")
    print(f"High:            ₹{latest['High']:.2f}")
    print(f"Low:             ₹{latest['Low']:.2f}")
    print(f"Volume:          {latest['Volume']:,.0f}")
    
    print("\n" + "─"*70)
    print("MOVING AVERAGES")
    print("─"*70)
    print(f"MA 5:            ₹{latest['MA_5']:.2f}")
    print(f"MA 20:           ₹{latest['MA_20']:.2f}")
    print(f"MA 50:           ₹{latest['MA_50']:.2f}")
    
    print("\n" + "─"*70)
    print("TECHNICAL INDICATORS")
    print("─"*70)
    print(f"RSI:             {latest['RSI']:.2f}")
    
    if latest['RSI'] < 30:
        print("                 → Oversold (Buy Signal)")
    elif latest['RSI'] > 70:
        print("                 → Overbought (Sell Signal)")
    else:
        print("                 → Neutral")
    
    print(f"MACD:            {latest['MACD']:.2f}")
    print(f"Signal Line:     {latest['Signal_Line']:.2f}")
    
    if latest['MACD'] > latest['Signal_Line']:
        print("                 → Bullish")
    else:
        print("                 → Bearish")
    
    print(f"BB Upper:        ₹{latest['BB_Upper']:.2f}")
    print(f"BB Lower:        ₹{latest['BB_Lower']:.2f}")
    
    # Train model and get signal
    predictor.train_models()
    signal_data = predictor.generate_signals()
    
    return predictor


# ============================================================================
# EXAMPLE 4: Custom Analysis Function
# ============================================================================

def analyze_stock_custom(ticker, period='1y', save_charts=True):
    """
    Custom function to analyze any stock
    
    Args:
        ticker (str): Stock symbol with .NS suffix
        period (str): Time period ('1y', '2y', '5y', 'max')
        save_charts (bool): Whether to save chart images
    
    Returns:
        dict: Analysis results
    """
    print(f"\nAnalyzing {ticker}...")
    
    try:
        predictor = IndianStockPredictor(ticker, period=period)
        predictor.fetch_data()
        predictor.calculate_technical_indicators()
        predictor.train_models()
        signal_data = predictor.generate_signals()
        
        if save_charts:
            predictor.plot_predictions(
                f'/mnt/user-data/outputs/{ticker.replace(".NS", "")}_predictions.png'
            )
            predictor.plot_technical_analysis(
                f'/mnt/user-data/outputs/{ticker.replace(".NS", "")}_technical.png'
            )
        
        return {
            'success': True,
            'ticker': ticker,
            'signal': signal_data['signal'],
            'data': signal_data
        }
        
    except Exception as e:
        return {
            'success': False,
            'ticker': ticker,
            'error': str(e)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("INDIAN STOCK MARKET PREDICTION - EXAMPLES")
    print("="*70)
    
    # Run Example 1: Single stock
    print("\n" + "🔍 Running Example 1...")
    predictor1, signal1 = example_single_stock()
    
    # Run Example 2: Multiple stocks
    print("\n" + "🔍 Running Example 2...")
    results2 = example_multiple_stocks()
    
    # Run Example 3: Detailed analysis
    print("\n" + "🔍 Running Example 3...")
    predictor3 = example_detailed_analysis()
    
    # Run Example 4: Custom function
    print("\n" + "🔍 Running Example 4...")
    print("\n\nEXAMPLE 4: Using Custom Analysis Function")
    print("="*70)
    
    custom_stocks = ['SBIN.NS', 'ITC.NS']
    for stock in custom_stocks:
        result = analyze_stock_custom(stock, period='6mo', save_charts=True)
        if result['success']:
            print(f"✅ {stock}: {result['signal']}")
        else:
            print(f"❌ {stock}: {result['error']}")
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED!")
    print("="*70)
    print("\nGenerated Files:")
    print("  - Prediction charts (predictions.png)")
    print("  - Technical analysis charts (technical_analysis.png)")
    print("\nReview the signals and charts before making any trading decisions!")
    print("⚠️  Remember: This is for educational purposes only, not financial advice!")
