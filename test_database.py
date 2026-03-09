from src.database import create_user, authenticate_user, get_user_trades, save_trade, update_user_balance
from datetime import datetime

print("="*60)
print("🧪 TESTING YASEN-ALPHA DATABASE")
print("="*60)

# Test user creation
print("\n📝 Creating test user...")
result = create_user("testuser", "test@example.com", "password123", 100.0)
if result:
    print("✅ User created successfully")
else:
    print("❌ User already exists (that's okay)")

# Test authentication
print("\n🔐 Testing authentication...")
user = authenticate_user("testuser", "password123")
if user:
    print(f"✅ Login successful!")
    print(f"   User ID: {user['id']}")
    print(f"   Username: {user['username']}")
    print(f"   Balance: ${user['current_balance']}")
else:
    print("❌ Login failed")

# Test saving a trade
if user:
    print("\n💰 Testing trade save...")
    trade_data = {
        'entry_date': datetime.now().isoformat(),
        'exit_date': datetime.now().isoformat(),
        'entry_price': 50000.0,
        'exit_price': 51000.0,
        'pnl_pct': 0.02,
        'pnl_usd': 20.0,
        'confidence': 0.65,
        'signal_type': 'BUY',
        'trade_type': 'test'
    }
    trade_id = save_trade(user['id'], trade_data)
    print(f"✅ Trade saved with ID: {trade_id}")
    
    # Test retrieving trades
    print("\n📊 Retrieving trades...")
    trades = get_user_trades(user['id'])
    print(f"Found {len(trades)} trades")
    
    # Test updating balance
    print("\n💵 Testing balance update...")
    update_user_balance(user['id'], 120.0)
    updated_user = authenticate_user("testuser", "password123")
    print(f"New balance: ${updated_user['current_balance']}")

print("\n" + "="*60)
print("✅ DATABASE TEST COMPLETE")
print("="*60)
