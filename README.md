# 🚀 YASEN-ALPHA ML Trading System

<div align="center">

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A production-grade Bitcoin prediction system with 59.19% win rate using ensemble machine learning**

[Features](#features) •
[Performance](#performance-metrics) •
[Quick Start](#quick-start) •
[API](#api) •
[Documentation](#documentation)

</div>

---

## 📑 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Performance Metrics](#performance-metrics)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Dashboard](#dashboard)
- [API](#api)
- [Telegram Bot](#telegram-bot)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Model Optimization](#model-optimization)
- [Environment Variables](#environment-variables)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments](#acknowledgments)
- [Disclaimer](#disclaimer)

---

## 📊 Overview

YASEN-ALPHA is a complete, production-ready cryptocurrency trading system that combines automated data pipelines, **78 engineered features**, and a **5-model XGBoost ensemble** achieving **59.19% win rate** over **8,209 backtested trades**. Built with production ML engineering practices, this system demonstrates end-to-end machine learning deployment from data collection to real-time API delivery.

---

## ✨ Features

- ✅ **Automated Data Pipeline** - Fetches live data from Kraken and Bitstamp exchanges
- ✅ **78 Engineered Features** - Technical indicators, temporal features, lagged values, rolling statistics
- ✅ **5-Model XGBoost Ensemble** - Weighted voting for robust predictions
- ✅ **Dynamic Thresholding** - Adapts to market volatility (0.47 optimal threshold)
- ✅ **Real-time API** - FastAPI backend with comprehensive documentation
- ✅ **Interactive Dashboard** - Streamlit UI with live signals and P&L tracking
- ✅ **Telegram Bot** - Instant signals via Telegram
- ✅ **Comprehensive Backtesting** - 8,209 trades validated
- ✅ **Docker Support** - Easy deployment anywhere
- ✅ **MLflow Integration** - Experiment tracking and model versioning

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Kraken  │  │ Bitstamp │  │  Coinbase│  │   CCXT   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       └─────────────┴──────────────┴─────────────┘         │
│                         │                                   │
│                         ▼                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │              FEATURE ENGINEERING                    │    │
│  │  • Price features (returns, log returns)           │    │
│  │  • Technical indicators (RSI, MACD, BB)            │    │
│  │  • Temporal features (hour, day, month)            │    │
│  │  • Lagged features (1-24h)                         │    │
│  │  • Rolling statistics (6-168h)                     │    │
│  │  • Sentiment indicators (Fear & Greed)             │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │              MODEL ENSEMBLE                         │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │    │
│  │  │ XGBoost 1│  │ XGBoost 2│  │ XGBoost 3│         │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘         │    │
│  │       └─────────────┴──────────────┘               │    │
│  │                    │                               │    │
│  │                    ▼                               │    │
│  │          ┌─────────────────┐                        │    │
│  │          │ Weighted Voting │                        │    │
│  │          └────────┬────────┘                        │    │
│  └───────────────────┼─────────────────────────────────┘    │
│                      │                                      │
│                      ▼                                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │               PRODUCTION SERVICES                   │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐   │    │
│  │  │   FastAPI  │  │  Streamlit │  │  Telegram  │   │    │
│  │  │   Backend  │  │  Dashboard │  │     Bot    │   │    │
│  │  └────────────┘  └────────────┘  └────────────┘   │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 📈 Performance Metrics

<div align="center">

| Metric | Value | Description |
|:-------|------:|:------------|
| **Win Rate** | 🏆 **59.19%** | Out of 8,209 backtested trades |
| **Total Trades** | 📊 **8,209** | Statistically significant sample |
| **Sharpe Ratio** | 📈 **1.24** | Excellent risk-adjusted returns |
| **Features** | 🔧 **78** | Engineered indicators |
| **Models** | 🤖 **5** | XGBoost ensemble |
| **Data History** | 📅 **9+ years** | 31,693 hourly samples |
| **Training Samples** | 📚 **31,470** | After feature engineering |
| **Best Threshold** | ⚖️ **0.47** | Optimized for maximum win rate |

</div>

---

## 🛠️ Tech Stack

| Category | Technologies |
|:---------|:-------------|
| **Languages** | Python 3.11 |
| **ML/AI** | XGBoost, LightGBM, Scikit-learn, Optuna |
| **Data Processing** | Pandas, NumPy, CCXT, Pandas-TA |
| **API** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit, Plotly |
| **Infrastructure** | Docker, Railway, Git, GitHub Actions |
| **Database** | PostgreSQL, Redis |
| **Monitoring** | MLflow, Loguru |
| **Testing** | Pytest, Pytest-cov |

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11 required
python --version  # Should show 3.11.x
```

### Installation

```bash
# Clone repository
git clone https://github.com/emyasenc/YASEN-ALPHA-ML-Trading-System.git
cd YASEN-ALPHA-ML-Trading-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys (optional for public data)
# - BINANCE_API_KEY=your_key_here
# - BINANCE_SECRET_KEY=your_secret_here
# - TELEGRAM_BOT_TOKEN=your_token
```

### Run Complete Pipeline

```bash
# Run all pipeline stages
python scripts/run_pipeline.py --stage all
```

**This will:**
- 📥 Fetch latest data from exchanges
- 🔧 Engineer 78 features
- 🤖 Train ensemble model
- 📊 Run backtest
- 🔮 Generate prediction

### Individual Stages

```bash
# Data collection only
python scripts/run_pipeline.py --stage data

# Feature engineering only
python scripts/run_pipeline.py --stage features

# Model training only
python scripts/run_pipeline.py --stage train

# Backtest only
python scripts/run_pipeline.py --stage backtest

# Get current signal
python scripts/run_pipeline.py --stage predict
```
📊 Dashboard
Launch Dashboard
bash
streamlit run src/dashboard/app.py
Open browser to http://localhost:8501

Dashboard Features
Feature	Description
Real-time signals	BUY/HOLD with confidence scores
Volatility indicators	Current market volatility
Price charts	Interactive Plotly charts with historical signals
Account tracking	P&L, balance history
Trade logging	Complete trade history with export
Performance metrics	Win rate, Sharpe ratio, drawdowns
🌐 API
Start API Server
bash
cd api
uvicorn main:app --reload --port 8000
API Documentation
bash
# Open in browser
http://localhost:8000/docs
Endpoints
Method	Endpoint	Description	Response
GET	/	API info	{"service": "YASEN-ALPHA API"}
GET	/health	Health check	{"status": "healthy"}
GET	/signal	Current trading signal	{"signal": "BUY/HOLD", "confidence": 0.72}
GET	/price	Latest BTC price	{"price": 62400, "change_24h": 2.3}
GET	/stats	Model statistics	{"win_rate": 0.5919, "trades": 8209}
GET	/history?days=30	Historical signals	[{"timestamp": "...", "signal": 1}]
Example Request
bash
curl -X GET "http://localhost:8000/signal" -H "accept: application/json"
Example Response
json
{
  "signal": "HOLD",
  "confidence": 0.125,
  "threshold_used": 0.47,
  "volatility": 0.0044,
  "timestamp": "2026-03-08T12:00:00"
}
🤖 Telegram Bot
Setup
bash
cd telegram_bot
pip install -r requirements.txt
python telegram_bot.py
Commands
Command	Description
/start	Welcome message with instructions
/signal	Current trading signal with confidence
/price	Latest BTC price and 24h change
/stats	Model performance statistics
/help	List of available commands
📁 Project Structure
text
├── src/                          # Source code
│   ├── data/                     # Data collection & validation
│   │   ├── sources/               # Exchange connectors
│   │   └── validation/             # Data quality checks
│   ├── features/                  # Feature engineering
│   │   └── builders/               # 78 feature generators
│   ├── models/                     # ML models
│   │   └── inference/               # Prediction service
│   ├── training/                   # Model training pipeline
│   ├── backtesting/                 # Performance validation
│   ├── risk/                        # Risk management
│   ├── execution/                   # Trade execution
│   ├── monitoring/                   # System monitoring
│   ├── api/                          # FastAPI backend
│   └── dashboard/                    # Streamlit frontend
├── scripts/                         # Automation scripts
│   ├── data/                         # Data collection scripts
│   ├── features/                      # Feature engineering scripts
│   ├── training/                       # Model training scripts
│   ├── backtesting/                     # Backtesting scripts
│   └── deployment/                      # Deployment scripts
├── config/                           # Configuration files
├── tests/                            # Unit & integration tests
│   ├── unit/                           # Unit tests
│   ├── integration/                     # Integration tests
│   └── fixtures/                        # Test fixtures
├── data/                             # Data storage (gitignored)
│   ├── raw/                            # Raw exchange data
│   ├── processed/                       # Feature-engineered data
│   └── models/                          # Trained models
├── logs/                              # Application logs
├── docs/                              # Documentation
├── notebooks/                         # Jupyter notebooks
├── requirements.txt                   # Dependencies
├── Dockerfile                         # Docker configuration
├── docker-compose.yml                  # Multi-container setup
└── README.md                           # This file
🧪 Testing
bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/ --cov-report=html

# Run specific test file
pytest tests/unit/test_data.py -v

# Run integration tests
pytest tests/integration/ -v
🐳 Docker Deployment
Build Image
bash
docker build -t yasen-alpha .
Run Container
bash
docker run -p 8000:8000 -p 8501:8501 yasen-alpha
Docker Compose
bash
docker-compose up -d
# Starts: API, Dashboard, PostgreSQL, Redis, MLflow
📈 Model Optimization
Hyperparameter Tuning
bash
python scripts/optimize_model_fast.py --trials 100
Feature Selection
bash
python scripts/analyze_features.py
Backtesting
bash
python scripts/run_backtest.py --model champion
🔐 Environment Variables
Create .env file:

bash
# Exchange API Keys (optional for public data)
BINANCE_API_KEY=your_key_here
BINANCE_SECRET_KEY=your_secret_here

# Database (optional)
DB_PASSWORD=your_password

# Telegram Bot (optional)
TELEGRAM_BOT_TOKEN=your_token

# Email Alerts (optional)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
📚 Documentation
API Reference

Data Pipeline

Feature Engineering

Model Training

Backtesting

Deployment

🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
Distributed under the MIT License. See LICENSE for more information.

👩‍💻 Author
Emma - CS Graduate, Machine Learning Engineer

GitHub: @emyasenc

LinkedIn: Emma Yasenc

Email: emma.yasenc@example.com

🙏 Acknowledgments
XGBoost for the amazing ML library

CCXT for cryptocurrency exchange connectivity

FastAPI for the lightning-fast API framework

Streamlit for the interactive dashboard

Optuna for hyperparameter optimization

⚠️ Disclaimer
IMPORTANT: This software is for educational and research purposes only.

📉 Cryptocurrency trading involves substantial risk of loss

📊 Past performance does NOT guarantee future results

❌ No guarantee of profitability

👨‍⚖️ Always consult with a qualified financial advisor

💰 Only trade with money you can afford to lose

The developers assume no responsibility for financial losses incurred through use of this software.

<div align="center">
Built with ❤️ for data science and machine learning

⬆ Back to Top

</div> ```