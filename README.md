# Interpretable Reinforcement Learning

A comprehensive framework for training and analyzing interpretable reinforcement learning agents with advanced visualization and evaluation capabilities.

## ⚠️ DISCLAIMER

**This project is for research and educational purposes only. XAI outputs may be unstable or misleading and should not be used for regulated decisions without human review.**

## Features

### Core Capabilities
- **Interpretable Q-Learning Agent**: Modern implementation with comprehensive analysis tools
- **Advanced Visualizations**: Interactive Q-table, policy, and value function visualizations
- **Comprehensive Evaluation**: Policy consistency, value convergence, and trajectory analysis
- **Interactive Demo**: Streamlit-based web interface for real-time exploration
- **Reproducible Research**: Deterministic seeding and configuration management

### Interpretability Methods
- **Q-Table Analysis**: Heatmap visualizations with interactive exploration
- **Policy Visualization**: Grid-based policy representation with action symbols
- **Value Function Analysis**: State value visualization and convergence tracking
- **Trajectory Analysis**: Episode efficiency and consistency metrics
- **Action Distribution**: Entropy and diversity analysis across actions
- **Policy Consistency**: Cross-episode action consistency evaluation

### Evaluation Metrics
- **Policy Consistency**: Measures action consistency across training episodes
- **Value Convergence**: Tracks Q-table stability and convergence patterns
- **Action Diversity**: Analyzes exploration vs exploitation balance
- **Trajectory Efficiency**: Evaluates reward-per-step efficiency
- **Overall Interpretability Score**: Weighted combination of all metrics

## 📁 Project Structure

```
├── src/interpretable_rl/          # Main package
│   ├── agents/                    # RL agent implementations
│   ├── visualizers/               # Visualization tools
│   ├── evaluators/                # Evaluation frameworks
│   └── config.py                  # Configuration management
├── demo/                          # Interactive Streamlit demo
├── configs/                       # Configuration files
├── data/                          # Data storage
├── assets/                        # Generated visualizations
├── models/                        # Saved models
├── tests/                         # Test suite
├── scripts/                       # Utility scripts
├── notebooks/                     # Jupyter notebooks
└── requirements.txt               # Dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/kryptologyst/Interpretable-Reinforcement-Learning.git
cd Interpretable-Reinforcement-Learning

# Install dependencies
pip install -r requirements.txt

# Run the main script
python 0740.py
```

### Development Setup
```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements.txt[dev]
```

## Quick Start

### Basic Usage
```python
from interpretable_rl import (
    InterpretableQLearningAgent,
    RLVisualizer,
    RLInterpretabilityEvaluator,
    RLConfig
)

# Create configuration
config = RLConfig(episodes=1000, learning_rate=0.1)

# Train agent
agent = InterpretableQLearningAgent(config)
agent.create_environment()
agent.initialize_q_table()
q_table, history = agent.train()

# Evaluate interpretability
evaluator = RLInterpretabilityEvaluator()
metrics = evaluator.evaluate_comprehensive(agent)

# Visualize results
visualizer = RLVisualizer(agent)
visualizer.save_all_plots()
```

### Interactive Demo
```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Usage Examples

### Training an Agent
```python
from interpretable_rl import InterpretableQLearningAgent, RLConfig

# Configure training
config = RLConfig(
    episodes=1000,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=0.1,
    random_seed=42
)

# Create and train agent
agent = InterpretableQLearningAgent(config)
agent.create_environment()
agent.initialize_q_table()
q_table, history = agent.train()
```

### Evaluating Interpretability
```python
from interpretable_rl import RLInterpretabilityEvaluator, EvaluationConfig

# Run comprehensive evaluation
eval_config = EvaluationConfig()
evaluator = RLInterpretabilityEvaluator(eval_config)
report = evaluator.generate_evaluation_report(agent)

print(f"Overall Interpretability Score: {report['overall_metrics']['overall_score']:.3f}")
```

### Creating Visualizations
```python
from interpretable_rl import RLVisualizer, VisualizationConfig

# Generate visualizations
viz_config = VisualizationConfig()
visualizer = RLVisualizer(agent, viz_config)

# Save all plots
visualizer.save_all_plots(save_dir="assets")

# Create interactive plots
interactive_q_table = visualizer.plot_interactive_q_table()
interactive_training = visualizer.plot_interactive_training_progress()
```

## Configuration

### RL Configuration
```python
from interpretable_rl import RLConfig

config = RLConfig(
    learning_rate=0.1,        # Learning rate for Q-learning
    discount_factor=0.99,      # Discount factor for future rewards
    episodes=1000,            # Number of training episodes
    epsilon=0.1,              # Initial exploration rate
    epsilon_decay=0.995,      # Epsilon decay rate
    epsilon_min=0.01,         # Minimum epsilon value
    random_seed=42,           # Random seed for reproducibility
    environment_name="FrozenLake-v1",  # Gymnasium environment
    is_slippery=False         # Environment slippiness
)
```

### Visualization Configuration
```python
from interpretable_rl import VisualizationConfig

viz_config = VisualizationConfig(
    style="seaborn-v0_8",     # Matplotlib style
    color_palette="viridis",  # Color palette
    figure_size=(10, 8),      # Default figure size
    dpi=300,                  # Plot resolution
    show_values=True,         # Show values on heatmaps
    q_table_cmap="RdYlBu_r",  # Q-table colormap
    policy_symbols=['←', '↓', '→', '↑']  # Policy symbols
)
```

### Evaluation Configuration
```python
from interpretable_rl import EvaluationConfig

eval_config = EvaluationConfig(
    consistency_window=100,           # Window for consistency analysis
    convergence_threshold=0.01,       # Convergence threshold
    convergence_window=50,           # Window for convergence analysis
    min_episodes_for_consistency=10   # Minimum episodes for analysis
)
```

## Evaluation Metrics

### Policy Consistency
- **Average Consistency**: Mean action consistency across states
- **Consistency Range**: Min/max consistency values
- **States Analyzed**: Number of states with sufficient data

### Value Function Convergence
- **Convergence Score**: Rate of Q-table stabilization
- **Stability Score**: Variance in recent Q-table changes
- **Convergence Rate**: Exponential decay fitting to changes

### Action Distribution
- **Action Entropy**: Shannon entropy of action distribution
- **Action Diversity**: 1 - max action probability
- **Distribution Analysis**: Per-action frequency analysis

### Trajectory Efficiency
- **Mean Efficiency**: Average reward per step
- **Length Consistency**: Variance in episode lengths
- **Success Rate**: Percentage of successful episodes

### Overall Interpretability Score
Weighted combination of all metrics:
- Policy Consistency: 30%
- Value Convergence: 25%
- Value Stability: 15%
- Action Diversity: 15%
- Trajectory Efficiency: 15%

## Visualizations

### Static Visualizations
- **Q-Table Heatmap**: State-action value visualization
- **Policy Grid**: Learned policy with action symbols
- **Value Function**: State value heatmap
- **Training Progress**: Multi-panel training metrics
- **Action Distribution**: Bar chart of action frequencies

### Interactive Visualizations
- **Interactive Q-Table**: Hover-enabled heatmap with Plotly
- **Training Dashboard**: Multi-subplot training progress
- **State Analysis**: State-specific Q-value exploration

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/interpretable_rl

# Run specific test file
pytest tests/test_agent.py
```

## API Reference

### InterpretableQLearningAgent
Main agent class for Q-learning with interpretability features.

**Methods:**
- `create_environment()`: Initialize gymnasium environment
- `initialize_q_table()`: Create Q-table with zeros
- `train()`: Train the agent using Q-learning
- `evaluate(num_episodes)`: Evaluate trained agent performance
- `get_policy()`: Extract greedy policy from Q-table
- `get_value_function()`: Extract value function from Q-table
- `analyze_policy_consistency()`: Analyze policy consistency
- `save_model(filepath)`: Save trained model
- `load_model(filepath)`: Load pre-trained model

### RLVisualizer
Comprehensive visualization tools for RL interpretability.

**Methods:**
- `plot_q_table()`: Q-table heatmap visualization
- `plot_policy()`: Policy grid visualization
- `plot_value_function()`: Value function heatmap
- `plot_training_progress()`: Multi-panel training metrics
- `plot_action_distribution()`: Action frequency analysis
- `plot_interactive_q_table()`: Interactive Q-table with Plotly
- `save_all_plots()`: Save all visualizations to files

### RLInterpretabilityEvaluator
Evaluation framework for RL interpretability metrics.

**Methods:**
- `evaluate_policy_consistency(agent)`: Policy consistency analysis
- `evaluate_value_convergence(agent)`: Value function convergence
- `evaluate_action_distribution(agent)`: Action distribution analysis
- `evaluate_trajectory_efficiency(agent)`: Trajectory efficiency metrics
- `evaluate_comprehensive(agent)`: Complete interpretability evaluation
- `generate_evaluation_report(agent)`: Detailed evaluation report

## 🔧 Development

### Code Style
- **Formatting**: Black for code formatting
- **Linting**: Ruff for code linting
- **Type Hints**: Full type annotation coverage
- **Documentation**: Google/NumPy style docstrings

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run pre-commit hooks
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct.

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the example notebooks

## Related Projects

- [Gymnasium](https://gymnasium.farama.org/): RL environment library
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/): RL algorithms
- [Captum](https://captum.ai/): Model interpretability
- [SHAP](https://shap.readthedocs.io/): Explainable AI

## Citation

If you use this project in your research, please cite:

```bibtex
@software{interpretable_rl,
  title={Interpretable Reinforcement Learning Framework},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Interpretable-Reinforcement-Learning}
}
```

---

**Remember**: This project is for research and educational purposes only. XAI outputs may be unstable or misleading and should not be used for regulated decisions without human review.
# Interpretable-Reinforcement-Learning
