# YASEN-ALPHA ML Trading System

**A production-grade ML trading infrastructure for cryptocurrency research and education.**

⚠️ **IMPORTANT DISCLAIMER: This is an educational project. The trading strategy demonstrated here is NOT profitable in live markets. Do not use for real trading.**

---

## 📑 Table of Contents

- [Overview](#overview)
- [What This Project Teaches](#what-this-project-teaches)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Disclaimer](#disclaimer)

---

## 📊 Overview

YASEN-ALPHA is a complete, production-ready cryptocurrency trading **infrastructure** that demonstrates end-to-end ML engineering:

- Automated data pipelines (Kraken + Bitstamp)
- 78 engineered features (technical indicators, temporal, lagged)
- 5-model XGBoost ensemble
- FastAPI backend with webhooks
- Streamlit dashboard
- Comprehensive backtesting framework

**What this is:** A learning resource for ML deployment, API development, and trading system architecture.

**What this is NOT:** A profitable trading strategy or financial advice.

---

## 🎓 What This Project Teaches

| Concept | Implementation |
|---------|----------------|
| Production ML pipelines | Automated data → features → training → deployment |
| Feature engineering | 78 technical indicators from raw price data |
| Model ensembles | Weighted voting with 5 XGBoost models |
| Backtesting validation | Walk-forward testing with fee modeling |
| API development | FastAPI with rate limiting, caching, webhooks |
| Docker deployment | Containerized microservices |
| Experiment tracking | MLflow integration |

---

## 🏗️ System Architecture
┌─────────────────────────────────────────────────────────────┐
│ DATA LAYER │
│ ┌──────────┐ ┌──────────┐ │
│ │ Kraken │ │ Bitstamp │ │
│ └────┬─────┘ └────┬─────┘ │
│ └─────────────┘ │
│ │ │
│ ▼ │
│ ┌────────────────────────────────────────────────────┐ │
│ │ FEATURE ENGINEERING │ │
│ │ • Price features (returns, log returns) │ │
│ │ • Technical indicators (RSI, MACD, BB) │ │
│ │ • Temporal features (hour, day, month) │ │
│ │ • Lagged features (1-24h) │ │
│ │ • Rolling statistics (6-168h) │ │
│ └────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌────────────────────────────────────────────────────┐ │
│ │ MODEL ENSEMBLE │ │
│ │ ┌──────────┐ ┌──────────┐ ┌──────────┐ │ │
│ │ │ XGBoost 1│ │ XGBoost 2│ │ XGBoost 3│ ... │ │
│ │ └────┬─────┘ └────┬─────┘ └────┬─────┘ │ │
│ │ └─────────────┴──────────────┘ │ │
│ │ │ │ │
│ │ ▼ │ │
│ │ ┌─────────────────┐ │ │
│ │ │ Weighted Voting │ │ │
│ │ └────────┬────────┘ │ │
│ └───────────────────┼─────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌────────────────────────────────────────────────────┐ │
│ │ PRODUCTION SERVICES │ │
│ │ ┌────────────┐ ┌────────────┐ ┌────────────┐ │ │
│ │ │ FastAPI │ │ Streamlit │ │ Telegram │ │ │
│ │ │ Backend │ │ Dashboard │ │ Bot │ │ │
│ │ └────────────┘ └────────────┘ └────────────┘ │ │
│ └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

text

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Languages | Python 3.11 |
| ML/AI | XGBoost, Scikit-learn |
| Data Processing | Pandas, NumPy, CCXT, Pandas-TA |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Infrastructure | Docker, Git, GitHub Actions |
| Monitoring | MLflow, Loguru |
| Testing | Pytest |

---

## 🚀 Quick Start

### Prerequisites

```bash
python --version  # Python 3.11+
Installation
bash
# Clone repository
git clone https://github.com/emyasenc/YASEN-ALPHA-ML-Trading-System.git
cd YASEN-ALPHA-ML-Trading-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Run Complete Pipeline
bash
python scripts/run_pipeline.py --stage all
This will:

Fetch latest data from exchanges

Engineer 78 features

Train ensemble model

Run backtest

Generate prediction

Individual Stages
bash
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
📊 Dashboard
Launch Dashboard
bash
streamlit run src/dashboard/app.py
Open browser to http://localhost:8501

Dashboard Features
Feature	Description
Real-time signals	BUY/HOLD with confidence scores
Volatility indicators	Current market volatility
Price charts	Interactive Plotly charts
Performance metrics	Win rate, Sharpe ratio, drawdowns
🌐 API
Start API Server
bash
cd api
uvicorn main:app --reload --port 8000
API Documentation
Open browser to http://localhost:8000/docs

Endpoints
Method	Endpoint	Description
GET	/	API info
GET	/health	Health check
GET	/signal	Current signal
GET	/price	Latest BTC price
GET	/stats	Model statistics
GET	/history?days=30	Historical signals
Example Request
bash
curl -X GET "http://localhost:8000/signal" -H "accept: application/json"
Example Response
json
{
  "signal": "HOLD",
  "confidence": 0.125,
  "volatility": 0.0044,
  "timestamp": "2026-03-08T12:00:00"
}
📁 Project Structure
text
├── src/                    # Source code
│   ├── data/               # Data collection & validation
│   ├── features/           # Feature engineering (78 features)
│   ├── models/             # ML models & inference
│   ├── training/           # Model training pipeline
│   ├── backtesting/        # Performance validation
│   ├── api/                # FastAPI backend
│   └── dashboard/          # Streamlit frontend
├── scripts/                # Automation scripts
├── config/                 # Configuration files
├── tests/                  # Unit & integration tests
├── data/                   # Data storage (gitignored)
├── logs/                   # Application logs
├── docs/                   # Documentation
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker configuration
└── README.md               # This file

🧪 Testing
bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/ --cov-report=html

# Run specific test file
pytest tests/unit/test_data.py -v
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
Starts: API, Dashboard, PostgreSQL, Redis, MLflow

📚 Documentation
API Reference

Data Pipeline

Feature Engineering

Model Training

Backtesting

Deployment

🤝 Contributing
Contributions are welcome!

Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
Distributed under the MIT License. See LICENSE for more information.

👩‍💻 Author
Emma Yasenchak - ML Engineer

GitHub: @emyasenc

LinkedIn: Emma Yasenchak

⚠️ Disclaimer
IMPORTANT: This software is for educational and research purposes only.

📉 Cryptocurrency trading involves substantial risk of loss

📊 Past performance does NOT guarantee future results

❌ No guarantee of profitability

👨‍⚖️ Always consult with a qualified financial advisor

💰 Only trade with money you can afford to lose

The developers assume no responsibility for financial losses incurred through use of this software.

Built with ❤️ for ML engineering education
