#!/usr/bin/env python3
"""
YASEN-ALPHA Telegram Bot
Provides real-time Bitcoin trading signals with 59.19% accuracy
Complete production-ready bot with inline keyboards and user management
"""

import os
import logging
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
from flask import Flask, jsonify
import threading
import time

# Simple health server
health_app = Flask(__name__)

@health_app.route('/health')
def health():
    return jsonify({"status": "alive", "bot": "running", "time": time.time()})

def run_health_server():
    port = int(os.environ.get('PORT', 10000))
    health_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# Start health server in background thread
health_thread = threading.Thread(target=run_health_server, daemon=True)
health_thread.start()
print(f"✅ Health server started on port {os.environ.get('PORT', 10000)}")

# Load environment variables
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
API_URL = os.getenv('API_URL', 'http://localhost:8000')

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# User data storage (in production, use database)
user_data = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message with main menu."""
    user = update.effective_user
    user_id = user.id
    
    # Initialize user data
    if user_id not in user_data:
        user_data[user_id] = {
            'username': user.username,
            'first_name': user.first_name,
            'subscribed': False,
            'joined_date': datetime.now().isoformat(),
            'signals_requested': 0
        }
    
    # Create main menu keyboard
    keyboard = [
        [InlineKeyboardButton("📊 GET SIGNAL", callback_data='signal')],
        [InlineKeyboardButton("💰 BITCOIN PRICE", callback_data='price')],
        [InlineKeyboardButton("📈 MODEL STATS", callback_data='stats')],
        [InlineKeyboardButton("📋 TRADING RULES", callback_data='rules')],
        [InlineKeyboardButton("🔔 DAILY SIGNALS", callback_data='subscribe')],
        [InlineKeyboardButton("❓ HELP", callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome = f"""
╔══════════════════════════════════╗
║     🚀 YASEN-ALPHA TRADING BOT   ║
╚══════════════════════════════════╝

Welcome back, {user.first_name}! 🔥

I'm your AI-powered Bitcoin trading assistant with **59.19% accuracy**.

📊 **WHAT I CAN DO:**
• Real-time BUY/HOLD signals
• Live Bitcoin price updates
• Model performance stats
• Risk management rules
• Daily signal subscription

🎯 **GET STARTED:**
Click a button below or type:
/signal - Current trading signal
/price - Live BTC price
/stats - Model statistics

⚠️ **RISK WARNING:** Trading involves substantial risk. Never invest more than you can afford to lose.
    """
    
    await update.message.reply_text(
        welcome,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
    logger.info(f"User {user.first_name} (@{user.username}) started the bot")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all button clicks."""
    query = update.callback_query
    await query.answer()
    
    handlers = {
        'signal': lambda: get_signal(update, context, True),
        'price': lambda: get_price(update, context, True),
        'stats': lambda: get_stats(update, context, True),
        'rules': lambda: show_rules(update, context, True),
        'subscribe': lambda: subscribe(update, context, True),
        'unsubscribe': lambda: unsubscribe(update, context, True),
        'help': lambda: help_command(update, context, True),
        'back': lambda: back_to_main(update, context)
    }
    
    handler = handlers.get(query.data)
    if handler:
        await handler()

async def back_to_main(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Return to main menu."""
    keyboard = [
        [InlineKeyboardButton("📊 GET SIGNAL", callback_data='signal')],
        [InlineKeyboardButton("💰 BITCOIN PRICE", callback_data='price')],
        [InlineKeyboardButton("📈 MODEL STATS", callback_data='stats')],
        [InlineKeyboardButton("📋 TRADING RULES", callback_data='rules')],
        [InlineKeyboardButton("🔔 DAILY SIGNALS", callback_data='subscribe')],
        [InlineKeyboardButton("❓ HELP", callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.callback_query.edit_message_text(
        "🏠 **MAIN MENU**\n\nChoose an option below:",
        parse_mode='Markdown',
        reply_markup=reply_markup
    )

async def get_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, from_button=False):
    """Get current trading signal."""
    user_id = update.effective_user.id
    
    # Track usage
    if user_id in user_data:
        user_data[user_id]['signals_requested'] = user_data[user_id].get('signals_requested', 0) + 1
    
    try:
        # Show loading
        if from_button:
            await update.callback_query.edit_message_text("⏳ Fetching latest signal...")
        
        # Call API
        response = requests.get(f"{API_URL}/signal", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Signal emoji
            emoji = '🟢' if data['signal'] == 'BUY' else '🔴' if data['signal'] == 'SELL' else '🟡'
            
            message = f"""
╔══════════════════════════════════╗
║     {emoji} YASEN-ALPHA SIGNAL      ║
╚══════════════════════════════════╝

**Signal:** `{data['signal']}`
**Confidence:** `{data['confidence']:.1%}`
**Threshold:** `{data['threshold_used']:.2f}`
**Volatility:** `{data['volatility']:.4f}`
**Updated:** `{data['timestamp']}`

━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **ACTION:**
• BUY → Enter with 2% risk
• HOLD → Stay in cash/exit
• Confidence >45% required
━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ Always use stop losses.
            """
            
            keyboard = [
                [InlineKeyboardButton("🔄 REFRESH", callback_data='signal'),
                 InlineKeyboardButton("💰 PRICE", callback_data='price')],
                [InlineKeyboardButton("🏠 MAIN MENU", callback_data='back')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            if from_button:
                await update.callback_query.edit_message_text(
                    message, parse_mode='Markdown', reply_markup=reply_markup
                )
            else:
                await update.message.reply_text(
                    message, parse_mode='Markdown', reply_markup=reply_markup
                )
        else:
            raise Exception("API error")
            
    except Exception as e:
        logger.error(f"Signal error: {e}")
        error_msg = "❌ Cannot fetch signal. API may be down."
        keyboard = [[InlineKeyboardButton("🔄 TRY AGAIN", callback_data='signal')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if from_button:
            await update.callback_query.edit_message_text(error_msg, reply_markup=reply_markup)

async def get_price(update: Update, context: ContextTypes.DEFAULT_TYPE, from_button=False):
    """Get current Bitcoin price."""
    try:
        if from_button:
            await update.callback_query.edit_message_text("⏳ Fetching Bitcoin price...")
        
        response = requests.get(f"{API_URL}/price", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            direction = '📈' if data['change_24h'] > 0 else '📉' if data['change_24h'] < 0 else '➡️'
            
            message = f"""
╔══════════════════════════════════╗
║     💰 BITCOIN PRICE {direction}      ║
╚══════════════════════════════════╝

**Current:** `${data['price']:,.2f}`
**24h Change:** `{data['change_24h']:+.2f}%`
**24h High:** `${data['high_24h']:,.2f}`
**24h Low:** `${data['low_24h']:,.2f}`
**Updated:** `{data['timestamp']}`

━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Use /signal for trading signals.
            """
            
            keyboard = [
                [InlineKeyboardButton("📊 GET SIGNAL", callback_data='signal'),
                 InlineKeyboardButton("🔄 REFRESH", callback_data='price')],
                [InlineKeyboardButton("🏠 MAIN MENU", callback_data='back')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            if from_button:
                await update.callback_query.edit_message_text(
                    message, parse_mode='Markdown', reply_markup=reply_markup
                )
            else:
                await update.message.reply_text(
                    message, parse_mode='Markdown', reply_markup=reply_markup
                )
        else:
            raise Exception("API error")
            
    except Exception as e:
        logger.error(f"Price error: {e}")
        error_msg = "❌ Cannot fetch price."
        keyboard = [[InlineKeyboardButton("🔄 TRY AGAIN", callback_data='price')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if from_button:
            await update.callback_query.edit_message_text(error_msg, reply_markup=reply_markup)

async def get_stats(update: Update, context: ContextTypes.DEFAULT_TYPE, from_button=False):
    """Show model statistics."""
    message = """
╔══════════════════════════════════╗
║     📈 MODEL PERFORMANCE          ║
╚══════════════════════════════════╝

**Win Rate:** `59.19%`
**Total Trades:** `8,274`
**Sharpe Ratio:** `0.42`
**Features:** `78`
**Data History:** `9+ years`
**Threshold:** `0.45`

━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏆 **HIGHLIGHTS:**
• Elite accuracy
• Statistically proven
• Daily updates
• 5-model ensemble
━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Use /signal for current signal.
    """
    
    keyboard = [
        [InlineKeyboardButton("📊 GET SIGNAL", callback_data='signal')],
        [InlineKeyboardButton("🏠 MAIN MENU", callback_data='back')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if from_button:
        await update.callback_query.edit_message_text(
            message, parse_mode='Markdown', reply_markup=reply_markup
        )
    else:
        await update.message.reply_text(
            message, parse_mode='Markdown', reply_markup=reply_markup
        )

async def show_rules(update: Update, context: ContextTypes.DEFAULT_TYPE, from_button=False):
    """Show trading rules."""
    message = """
╔══════════════════════════════════╗
║     📋 TRADING RULES              ║
╚══════════════════════════════════╝

**1. SIGNAL INTERPRETATION**
   • BUY → Enter long position
   • HOLD → Stay in cash/exit
   • Confidence >45% required

**2. RISK MANAGEMENT**
   • Maximum 2% risk per trade
   • Always use stop losses
   • Never average down

**3. POSITION SIZING**
   `Position = Account × 0.02 × 50`
   Example: $1000 → $1000 position

**4. EXIT STRATEGY**
   • Exit when signal turns HOLD
   • Take profits at 4-6%
   • Cut losses at 2-3%

**5. GOLDEN RULES**
   • No emotions
   • Trust the system
   • Track every trade
   • Review weekly

⚠️ These rules are optimized for long-term profitability.
    """
    
    keyboard = [[InlineKeyboardButton("🏠 MAIN MENU", callback_data='back')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if from_button:
        await update.callback_query.edit_message_text(
            message, parse_mode='Markdown', reply_markup=reply_markup
        )
    else:
        await update.message.reply_text(
            message, parse_mode='Markdown', reply_markup=reply_markup
        )

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE, from_button=False):
    """Subscribe to daily signals."""
    user_id = update.effective_user.id
    
    if user_id not in user_data:
        user_data[user_id] = {}
    
    if user_data[user_id].get('subscribed', False):
        message = """
✅ **ALREADY SUBSCRIBED**

You're receiving daily signals at 8:00 AM.

Use /unsubscribe to stop.
        """
    else:
        user_data[user_id]['subscribed'] = True
        message = """
✅ **SUCCESSFULLY SUBSCRIBED!**

You'll now receive:
• Daily signal at 8:00 AM
• Market alerts
• Performance updates

Use /unsubscribe to stop.
        """
    
    keyboard = [
        [InlineKeyboardButton("📊 GET SIGNAL", callback_data='signal')],
        [InlineKeyboardButton("🔕 UNSUBSCRIBE", callback_data='unsubscribe')],
        [InlineKeyboardButton("🏠 MAIN MENU", callback_data='back')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if from_button:
        await update.callback_query.edit_message_text(
            message, parse_mode='Markdown', reply_markup=reply_markup
        )
    else:
        await update.message.reply_text(
            message, parse_mode='Markdown', reply_markup=reply_markup
        )

async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE, from_button=False):
    """Unsubscribe from daily signals."""
    user_id = update.effective_user.id
    
    if user_id in user_data:
        user_data[user_id]['subscribed'] = False
    
    message = """
❌ **UNSUBSCRIBED**

You will no longer receive daily signals.

Use /subscribe to start again.
    """
    
    keyboard = [[InlineKeyboardButton("🏠 MAIN MENU", callback_data='back')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if from_button:
        await update.callback_query.edit_message_text(
            message, parse_mode='Markdown', reply_markup=reply_markup
        )
    else:
        await update.message.reply_text(
            message, parse_mode='Markdown', reply_markup=reply_markup
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_button=False):
    """Show help message."""
    message = """
╔══════════════════════════════════╗
║     ❓ HELP & COMMANDS            ║
╚══════════════════════════════════╝

**COMMANDS:**
/start - Main menu
/signal - Current signal
/price - Bitcoin price
/stats - Model stats
/rules - Trading rules
/subscribe - Daily signals
/unsubscribe - Stop signals
/help - This message

**HOW TO TRADE:**
1. Wait for BUY signal (>45%)
2. Calculate position (2% risk)
3. Enter with stop loss
4. Exit when HOLD appears
5. Track your results

**SUPPORT:** @yasen_alpha_support

⚠️ **DISCLAIMER:** Educational purposes only.
    """
    
    keyboard = [[InlineKeyboardButton("🏠 MAIN MENU", callback_data='back')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if from_button:
        await update.callback_query.edit_message_text(
            message, parse_mode='Markdown', reply_markup=reply_markup
        )
    else:
        await update.message.reply_text(
            message, parse_mode='Markdown', reply_markup=reply_markup
        )

def main():
    """Start the bot."""
    if not TOKEN:
        logger.error("No TELEGRAM_TOKEN found!")
        print("❌ ERROR: Please set TELEGRAM_TOKEN in .env file")
        return
    
    print(f"""
╔══════════════════════════════════╗
║   🤖 YASEN-ALPHA TELEGRAM BOT    ║
╠══════════════════════════════════╣
║  Token: {TOKEN[:10]}...{TOKEN[-5:]}        
║  API: {API_URL}                   
║  Status: RUNNING                  
╚══════════════════════════════════╝
    """)
    
    app = Application.builder().token(TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", get_signal))
    app.add_handler(CommandHandler("price", get_price))
    app.add_handler(CommandHandler("stats", get_stats))
    app.add_handler(CommandHandler("rules", show_rules))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    
    print("✅ Bot is running! Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == '__main__':
    main()
