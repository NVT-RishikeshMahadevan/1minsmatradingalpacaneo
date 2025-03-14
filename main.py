import streamlit as st
import yfinance as yf
import pandas as pd
import time
import plotly.graph_objects as go
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
import json
import os

# Set page config
st.set_page_config(
    page_title="BTC-USD SMA Trading Bot",
    page_icon="📈",
    layout="wide"
)

# File to store bot state
STATE_FILE = "sma_bot_state.json"

def load_bot_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        return {
            "is_active": False, 
            "last_run": None, 
            "position": None,
            "entry_price": None,
            "position_size": None
        }
    except:
        return {
            "is_active": False, 
            "last_run": None, 
            "position": None,
            "entry_price": None,
            "position_size": None
        }

def save_bot_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

@st.cache_resource(ttl=60)
def create_trading_client():
    api_key = "PK25YZNBYBYX0XQJNK5A"
    secret_key = "CfZx1CtNITOdYKpwxVYCec02k6WBT0EJBYSS5WgZ"
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=True)

def place_market_order(symbol, qty, side):
    client = create_trading_client()
    try:
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC,
            type=OrderType.MARKET
        )
        order = client.submit_order(order_request)
        return True, f"Order placed successfully: {side.value.title()} {qty} {symbol}"
    except Exception as e:
        return False, f"Error placing order: {str(e)}"

def fetch_crypto_data():
    """Fetch the last 60 minutes of BTC-USD data"""
    btc = yf.Ticker("BTC-USD")
    data = btc.history(period="1d", interval="1m")
    return data

def create_candlestick_chart(df):
    """Create a candlestick chart using plotly"""
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='BTC-USD'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA50'],
        line=dict(color='orange', width=2),
        name='SMA-50'
    ))
    
    fig.update_layout(
        title='BTC-USD Price Movement with SMA-50',
        yaxis_title='Price (USD)',
        xaxis_title='Time',
        template='plotly_dark'
    )
    return fig

def calculate_position_pl(entry_price, current_price, position_size):
    """Calculate P/L for current position"""
    try:
        if entry_price is not None and position_size is not None and current_price is not None:
            pl = (current_price - entry_price) * position_size
            pl_pct = (current_price - entry_price) / entry_price * 100
            return pl, pl_pct
    except Exception as e:
        st.error(f"Error calculating P/L: {str(e)}")
    return 0.0, 0.0

# Initialize session state for update counter
if 'update_counter' not in st.session_state:
    st.session_state.update_counter = 0

# Load bot state
bot_state = load_bot_state()

# Title
st.title("BTC-USD SMA Trading Bot")
st.markdown("*Trading based on SMA-50 crossover strategy*")

# Top Controls Section
st.markdown("### Bot Controls")
control_col1, control_col2, control_col3, control_col4 = st.columns([1, 1, 1, 1])

# Bot Start/Stop Controls
with control_col1:
    if not bot_state["is_active"]:
        if st.button("▶️ Start Bot", use_container_width=True):
            bot_state["is_active"] = True
            bot_state["last_run"] = None
            save_bot_state(bot_state)
            st.cache_data.clear()
            st.rerun()
    else:
        if st.button("⏹️ Stop Bot", use_container_width=True):
            bot_state["is_active"] = False
            save_bot_state(bot_state)
            st.cache_data.clear()
            st.rerun()

# Bot Status
with control_col2:
    if bot_state["is_active"]:
        st.success("Bot Status: Running ✅")
    else:
        st.error("Bot Status: Stopped ⛔")

# Current Position
with control_col3:
    position_status = bot_state["position"].title() if bot_state["position"] else "No Position"
    st.info(f"Current Position: {position_status}")

# Close Positions/Cancel Orders
with control_col4:
    if st.button("🔄 Reset Positions/Orders", use_container_width=True):
        client = create_trading_client()
        try:
            client.close_all_positions(cancel_orders=True)
            # Reset all position-related state
            bot_state["position"] = None
            bot_state["entry_price"] = None
            bot_state["position_size"] = None
            bot_state["last_run"] = None
            save_bot_state(bot_state)
            st.success("All positions closed and orders cancelled")
            time.sleep(2)
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Error resetting positions: {str(e)}")

# Separator
st.markdown("---")

# Create placeholders for dynamic content
chart_placeholder = st.empty()
metrics_placeholder = st.empty()
table_placeholder = st.empty()
positions_header_placeholder = st.empty()
positions_placeholder = st.empty()
metrics_placeholder2 = st.empty()
info_placeholder = st.empty()
timer_placeholder = st.empty()
bot_info_placeholder = st.empty()

# Main app loop that will refresh
while bot_state["is_active"]:
    st.session_state.update_counter += 1
    
    # Fetch data
    df = fetch_crypto_data()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # Trading logic
    current_time = datetime.now()
    last_run = datetime.fromisoformat(bot_state["last_run"]) if bot_state["last_run"] else None
    time_since_last_run = (current_time - last_run).total_seconds() if last_run else float('inf')
    
    if time_since_last_run >= 60:
        current_price = df['Close'].iloc[-1]
        current_sma = df['SMA50'].iloc[-1]
        
        # Trading signals
        if current_price > current_sma and bot_state["position"] != "long":
            success, message = place_market_order("BTCUSD", 0.01, OrderSide.BUY)
            if success:
                # Initialize all position-related state
                bot_state["position"] = "long"
                bot_state["entry_price"] = float(current_price)  # Convert to float
                bot_state["position_size"] = 0.01
                save_bot_state(bot_state)
                st.success(f"Buy signal executed at {current_time.strftime('%H:%M:%S')} - Entry Price: ${current_price:,.2f}")
            else:
                st.error(message)
        
        elif current_price < current_sma and bot_state["position"] == "long":
            success, message = place_market_order("BTCUSD", 0.01, OrderSide.SELL)
            if success:
                pl, pl_pct = calculate_position_pl(
                    float(bot_state["entry_price"]), 
                    float(current_price), 
                    float(bot_state["position_size"])
                )
                # Reset all position-related state
                bot_state["position"] = None
                bot_state["entry_price"] = None
                bot_state["position_size"] = None
                save_bot_state(bot_state)
                st.success(f"Sell signal executed at {current_time.strftime('%H:%M:%S')} - P/L: ${pl:,.2f} ({pl_pct:.2f}%)")
            else:
                st.error(message)
        
        bot_state["last_run"] = current_time.isoformat()
        save_bot_state(bot_state)
    
    # Create and display chart
    fig = create_candlestick_chart(df)
    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"btc_chart_{st.session_state.update_counter}")
    
    # Display metrics
    latest_price = df['Close'].iloc[-1]
    price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
    price_change_pct = (price_change / df['Close'].iloc[0]) * 100
    
    with metrics_placeholder.container():
        col1, col2 = st.columns(2)
        col1.metric(
            "Current Price",
            f"${latest_price:,.2f}"
        )
        col2.metric(
            "Price Change (1h)",
            f"${price_change:,.2f}",
            f"{price_change_pct:.2f}%"
        )
    
    # Display table
    with table_placeholder.container():
        st.subheader("Last 10 Minutes Candlestick Data")
        df_display = df.tail(10).copy()
        df_display = df_display.iloc[::-1]
        df_display.index = df_display.index.strftime('%H:%M:%S')
        df_display = df_display.round(2)
        st.dataframe(
            df_display[['Open', 'High', 'Low', 'Close', 'SMA50']], 
            use_container_width=True,
            key=f"price_table_{st.session_state.update_counter}"
        )
    
    # Position Management Section
    positions_header_placeholder.header("Current Positions")
    
    try:
        positions_data = []
        current_price = float(df['Close'].iloc[-1])
        client = create_trading_client()
        
        # Get positions from Alpaca
        alpaca_positions = client.get_all_positions()
        
        # Get recent orders to find entry price
        orders_request = GetOrdersRequest(
            status="closed",  # Using string value instead of enum
            limit=100
        )
        recent_orders = client.get_orders(orders_request)
        
        # Process positions from Alpaca
        for position in alpaca_positions:
            if position.symbol == "BTCUSD":
                # Find the most recent buy order for entry price
                entry_order = None
                for order in recent_orders:
                    if order.symbol == "BTCUSD" and order.side.value == "buy":
                        entry_order = order
                        break
                
                side = "Long" if float(position.qty) > 0 else "Short"
                position_size = abs(float(position.qty))
                current_price = float(position.current_price)
                entry_price = float(entry_order.filled_avg_price) if entry_order else float(position.avg_entry_price)
                market_value = float(position.market_value)
                unrealized_pl = float(position.unrealized_pl)
                pl_pct = (unrealized_pl / (entry_price * position_size)) * 100
                
                positions_data.append({
                    "Symbol": position.symbol,
                    "Side": side,
                    "Quantity": position_size,
                    "Market Value": f"${market_value:,.2f}",
                    "Average Entry": f"${entry_price:,.2f}",
                    "Current Price": f"${current_price:,.2f}",
                    "Unrealized P/L": f"${unrealized_pl:,.2f}",
                    "P/L %": f"{pl_pct:.2f}%"
                })
                
                # Update bot state to match Alpaca position
                bot_state["position"] = side.lower()
                bot_state["entry_price"] = entry_price
                bot_state["position_size"] = position_size
                save_bot_state(bot_state)
        
        if positions_data:
            positions_placeholder.table(pd.DataFrame(positions_data))
            
            # Calculate and display performance metrics
            try:
                position = positions_data[0]  # We only trade one position
                total_value = float(position['Market Value'].replace('$', '').replace(',', ''))
                total_pl = float(position['Unrealized P/L'].replace('$', '').replace(',', ''))
                pl_pct = float(position['P/L %'].replace('%', ''))
                
                # Display metrics in columns
                with metrics_placeholder2.container():
                    met_col1, met_col2, met_col3 = st.columns(3)
                    met_col1.metric(
                        "Total Portfolio Value",
                        f"${total_value:,.2f}"
                    )
                    met_col2.metric(
                        "Total Unrealized P/L",
                        f"${total_pl:,.2f}"
                    )
                    met_col3.metric(
                        "Return %",
                        f"{pl_pct:.2f}%"
                    )
            except Exception as e:
                metrics_placeholder2.error(f"Error calculating metrics: {str(e)}")
        else:
            positions_placeholder.info("No open positions")
            # Reset bot state if no positions found
            bot_state["position"] = None
            bot_state["entry_price"] = None
            bot_state["position_size"] = None
            save_bot_state(bot_state)
                
    except Exception as e:
        positions_placeholder.error(f"Error updating positions: {str(e)}")
    
    # Auto refresh info
    info_placeholder.markdown("---")
    info_placeholder.markdown("### Auto-refresh Status")
    
    # Calculate time until next run
    if bot_state["last_run"]:
        last_run_time = datetime.fromisoformat(bot_state["last_run"])
        next_run_time = last_run_time + timedelta(seconds=60)
        time_until_next = max(0, (next_run_time - datetime.now()).total_seconds())
        
        timer_placeholder.info(f"""
        Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Last trade check: {last_run_time.strftime('%Y-%m-%d %H:%M:%S')}
        Next trade check: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')} ({int(time_until_next)} seconds remaining)
        """)
    else:
        timer_placeholder.info(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Waiting for first trade check.")
    
    # Add bot information section at the bottom
    bot_info_placeholder.markdown("---")
    bot_info_placeholder.subheader("Bot Information")
    bot_info_placeholder.markdown("""
    This bot uses a simple SMA-50 crossover strategy:
    - **Buy Signal**: When price crosses above SMA-50
    - **Sell Signal**: When price crosses below SMA-50
    
    Trading is executed through Alpaca's paper trading API.
    """)
    
    # Sleep for a short time to prevent excessive updates
    time.sleep(5)
    
    # Check if bot is still active before refreshing
    bot_state = load_bot_state()
    if not bot_state["is_active"]:
        break
        
# If bot is not active, display static dashboard
if not bot_state["is_active"]:
    # Fetch data
    df = fetch_crypto_data()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # Create and display chart
    fig = create_candlestick_chart(df)
    chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    # Display metrics
    latest_price = df['Close'].iloc[-1]
    price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
    price_change_pct = (price_change / df['Close'].iloc[0]) * 100
    
    with metrics_placeholder.container():
        col1, col2 = st.columns(2)
        col1.metric(
            "Current Price",
            f"${latest_price:,.2f}"
        )
        col2.metric(
            "Price Change (1h)",
            f"${price_change:,.2f}",
            f"{price_change_pct:.2f}%"
        )
    
    # Display table
    with table_placeholder.container():
        st.subheader("Last 10 Minutes Candlestick Data")
        df_display = df.tail(10).copy()
        df_display = df_display.iloc[::-1]
        df_display.index = df_display.index.strftime('%H:%M:%S')
        df_display = df_display.round(2)
        st.dataframe(
            df_display[['Open', 'High', 'Low', 'Close', 'SMA50']], 
            use_container_width=True
        )
    
    # Position Management Section
    positions_header_placeholder.header("Current Positions")
    
    try:
        positions_data = []
        client = create_trading_client()
        
        # Get positions from Alpaca
        alpaca_positions = client.get_all_positions()
        
        # Get recent orders to find entry price
        orders_request = GetOrdersRequest(
            status="closed",
            limit=100
        )
        recent_orders = client.get_orders(orders_request)
        
        # Process positions from Alpaca
        for position in alpaca_positions:
            if position.symbol == "BTCUSD":
                # Find the most recent buy order for entry price
                entry_order = None
                for order in recent_orders:
                    if order.symbol == "BTCUSD" and order.side.value == "buy":
                        entry_order = order
                        break
                
                side = "Long" if float(position.qty) > 0 else "Short"
                position_size = abs(float(position.qty))
                current_price = float(position.current_price)
                entry_price = float(entry_order.filled_avg_price) if entry_order else float(position.avg_entry_price)
                market_value = float(position.market_value)
                unrealized_pl = float(position.unrealized_pl)
                pl_pct = (unrealized_pl / (entry_price * position_size)) * 100
                
                positions_data.append({
                    "Symbol": position.symbol,
                    "Side": side,
                    "Quantity": position_size,
                    "Market Value": f"${market_value:,.2f}",
                    "Average Entry": f"${entry_price:,.2f}",
                    "Current Price": f"${current_price:,.2f}",
                    "Unrealized P/L": f"${unrealized_pl:,.2f}",
                    "P/L %": f"{pl_pct:.2f}%"
                })
        
        if positions_data:
            positions_placeholder.table(pd.DataFrame(positions_data))
            
            # Calculate and display performance metrics
            try:
                position = positions_data[0]  # We only trade one position
                total_value = float(position['Market Value'].replace('$', '').replace(',', ''))
                total_pl = float(position['Unrealized P/L'].replace('$', '').replace(',', ''))
                pl_pct = float(position['P/L %'].replace('%', ''))
                
                # Display metrics in columns
                with metrics_placeholder2.container():
                    met_col1, met_col2, met_col3 = st.columns(3)
                    met_col1.metric(
                        "Total Portfolio Value",
                        f"${total_value:,.2f}"
                    )
                    met_col2.metric(
                        "Total Unrealized P/L",
                        f"${total_pl:,.2f}"
                    )
                    met_col3.metric(
                        "Return %",
                        f"{pl_pct:.2f}%"
                    )
            except Exception as e:
                metrics_placeholder2.error(f"Error calculating metrics: {str(e)}")
        else:
            positions_placeholder.info("No open positions")
                
    except Exception as e:
        positions_placeholder.error(f"Error updating positions: {str(e)}")
    
    # Add bot information section at the bottom
    bot_info_placeholder.markdown("---")
    bot_info_placeholder.subheader("Bot Information")
    bot_info_placeholder.markdown("""
    This bot uses a simple SMA-50 crossover strategy:
    - **Buy Signal**: When price crosses above SMA-50
    - **Sell Signal**: When price crosses below SMA-50
    
    Trading is executed through Alpaca's paper trading API.
    """)
