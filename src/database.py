"""
YASEN-ALPHA Database Module
SQLite-based storage for multi-user support
"""

import sqlite3
import json
from datetime import datetime
import os
from pathlib import Path

DB_PATH = Path('data/yasen_alpha.db')

def init_database():
    """Initialize SQLite database with all tables."""
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            initial_balance REAL DEFAULT 100.0,
            current_balance REAL DEFAULT 100.0,
            settings TEXT DEFAULT '{}'
        )
    ''')
    
    # Trades table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            entry_date TEXT NOT NULL,
            exit_date TEXT,
            entry_price REAL NOT NULL,
            exit_price REAL,
            pnl_pct REAL,
            pnl_usd REAL,
            confidence REAL,
            signal_type TEXT,
            trade_type TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Positions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            entry_date TEXT,
            entry_price REAL,
            size REAL,
            confidence REAL,
            signal_type TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized at data/yasen_alpha.db")

def hash_password(password):
    """Simple password hashing."""
    import hashlib
    return hashlib.sha256(f"yasen_salt_{password}".encode()).hexdigest()

def create_user(username, email, password, initial_balance=100):
    """Register a new user."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, created_at, initial_balance, current_balance)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            username, 
            email, 
            hash_password(password),
            datetime.now().isoformat(),
            initial_balance,
            initial_balance
        ))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    """Check login credentials."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, email, initial_balance, current_balance 
        FROM users WHERE username = ? AND password_hash = ?
    ''', (username, hash_password(password)))
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return {
            'id': user[0],
            'username': user[1],
            'email': user[2],
            'initial_balance': user[3],
            'current_balance': user[4]
        }
    return None

def get_user_trades(user_id):
    """Get all trades for a user."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM trades WHERE user_id = ? ORDER BY entry_date DESC
    ''', (user_id,))
    
    columns = [description[0] for description in cursor.description]
    trades = []
    for row in cursor.fetchall():
        trade = dict(zip(columns, row))
        # Convert dates to strings for JSON serialization
        trades.append(trade)
    
    conn.close()
    return trades

def save_trade(user_id, trade_data):
    """Save a new trade."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO trades (
            user_id, entry_date, exit_date, entry_price, exit_price,
            pnl_pct, pnl_usd, confidence, signal_type, trade_type, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        trade_data['entry_date'],
        trade_data.get('exit_date'),
        trade_data['entry_price'],
        trade_data.get('exit_price'),
        trade_data.get('pnl_pct'),
        trade_data.get('pnl_usd'),
        trade_data.get('confidence', 0.5),
        trade_data.get('signal_type', 'MANUAL'),
        trade_data.get('trade_type', 'auto'),
        datetime.now().isoformat()
    ))
    
    conn.commit()
    trade_id = cursor.lastrowid
    conn.close()
    return trade_id

def update_user_balance(user_id, new_balance):
    """Update user's current balance."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE users SET current_balance = ? WHERE id = ?
    ''', (new_balance, user_id))
    
    conn.commit()
    conn.close()

def save_position(user_id, position_data):
    """Save or update open position."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Delete existing position if any
    cursor.execute('DELETE FROM positions WHERE user_id = ?', (user_id,))
    
    if position_data:
        cursor.execute('''
            INSERT INTO positions (user_id, entry_date, entry_price, size, confidence, signal_type, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            position_data['entry_date'],
            position_data['entry_price'],
            position_data['size'],
            position_data['confidence'],
            position_data['signal_type'],
            datetime.now().isoformat()
        ))
    
    conn.commit()
    conn.close()

def get_position(user_id):
    """Get user's open position."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM positions WHERE user_id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        columns = ['id', 'user_id', 'entry_date', 'entry_price', 'size', 'confidence', 'signal_type', 'updated_at']
        return dict(zip(columns, row))
    return None

def delete_position(user_id):
    """Delete user's open position."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute('DELETE FROM positions WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()

# Initialize database on import
init_database()
