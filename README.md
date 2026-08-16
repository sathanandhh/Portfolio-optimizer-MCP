# QuantPortfolioMCP: Research-Grade Portfolio Optimization

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

QuantPortfolioMCP is a mathematical powerhouse for portfolio construction. Moving beyond classical Markowitz, it includes robust estimators (Ledoit-Wolf shrinkage), machine learning clustering (Hierarchical Risk Parity), and downside-risk optimization (CVaR). It includes a full attribution layer for risk decomposition, designed to run flawlessly via standard input/output (`stdio` with Claude Desktop).

## Installation

```bash
git clone https://github.com/yourusername/QuantPortfolioMCP.git
cd QuantPortfolioMCP
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python mcpserver.py
```

