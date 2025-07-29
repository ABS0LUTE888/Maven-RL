<img width="1280" height="640" alt="github" src="https://github.com/user-attachments/assets/a1f61606-fcfc-4928-8a46-166f2c6d47b3" />

# Maven-RL

Maven-RL is a research-focused reinforcement learning framework for developing and testing trading agents in simulated
environments. Developed by **Ilya (Ilia) Shilov** and **Viacheslav Pomeshchikov**.

## Important Disclaimer

🚫 **This project is for research and educational purposes only**  
🚫 **Do not use with real money or real trading accounts**  
🚫 **The authors assume no responsibility for any financial losses incurred**  
🚫 **Trading involves significant risk and may result in loss of capital**

## Key Features

- **Reinforcement Learning Environment**: Custom Gymnasium environment for realistic trading simulation
- **Flexible Configuration**: Hydra-powered YAML configuration system
- **Diverse Reward Functions**: Multiple reward functions including risk-adjusted returns
- **Action Masking**: Prevents invalid trading actions
- **Comprehensive Metrics**: Detailed trading performance evaluation
- **Modular Architecture**: Easily extendable components
- **Custom Feature Extractor**: TCN feature extractor for time-series data

## Inspiration

Maven-RL draws inspiration from:

- Open-source projects: TensorTrade, FinRL®
- Research papers: DeepScalper, and others
- RL applications in quantitative finance

## Project Structure

```
Maven-RL/
├── configs/               # Configuration files
│   └── ...
├── env/                   # Trading environment
│   ├── core.py            # Core trading environment
│   ├── factory.py         # Environment creation
│   ├── masks.py           # Action masking functions
│   ├── metrics.py         # Performance metrics
│   ├── observations.py    # Observation builders
│   └── rewards.py         # Reward functions
├── feature_extractors/    # Feature extractors
│   ├── base.py            # Feature extractor registry
│   └── tcn.py             # TCN Feature Extractor
└── train.py               # Training script
```

## Getting Started

### Installation

1. Clone the repository:

```bash
git clone https://github.com/ABS0LUTE888/Maven-RL.git
cd Maven-RL
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Maven-RL uses Hydra for configuration management. Key configuration files include:

- **Environment Settings** (`configs/env`)

- **Algorithm Settings** (`configs/algo`)

- **Reward Functions** (`configs/reward`)

- **Observation Builders** (`configs/observations`)

- **Feature Extractors** (`configs/feature_extractor`)

- **Action Masking** (`configs/mask`)

- **Performance Metrics** (`configs/metrics`)

- **Paths** (`configs/paths`)

## Training an Agent

To train a trading agent using the default configuration:

```bash
python train.py
```

To train with custom configurations:

```bash
python train.py algo.learning_rate=1e-4 reward=equity_delta
```

Or you can simply edit `configs/default.yaml`

Dataset Requirements:

- Must be a `.pkl` file containing `List[pd.DataFrame]`
- Required columns:
    - One-hot asset encoding (`asset_{i}`)
    - Scaled prices (`1m_z_open`, `1m_z_high`, `1m_z_low`, `1m_z_close`)
    - Unscaled closing price (`1m_close`)
    - Scaled indicators (`1m_{indicator}`, `5m_{indicator}`). We use crypto data, but you can feel free to use
      anything.

## Future

We're actively working on:

- Open source dataset management and backtesting scripts
- Additional agents (A2C, DQN) and feature extractors
- Model-based RL integration
- Support for different trading strategies
- Enhanced modularity and customization
- Expanded data sources (e.g., LOB and sentiment data)

_(P.S. Since Maven-RL is built on our private codebase, it follows its own, separate update system;
current version is far from being perfect/production ready)_

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
