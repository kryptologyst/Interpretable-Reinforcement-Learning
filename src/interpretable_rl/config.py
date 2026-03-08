"""
Configuration management for Interpretable RL project.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from omegaconf import OmegaConf


@dataclass
class RLConfig:
    """Configuration for RL training parameters."""
    # Training parameters
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    episodes: int = 1000
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    
    # Environment parameters
    environment_name: str = "FrozenLake-v1"
    is_slippery: bool = False
    
    # Reproducibility
    random_seed: int = 42
    
    # Evaluation parameters
    eval_episodes: int = 100
    eval_frequency: int = 100
    
    # Visualization parameters
    save_plots: bool = True
    plot_dpi: int = 300
    figure_size: tuple = (10, 8)
    
    # Paths
    data_dir: str = "data"
    assets_dir: str = "assets"
    model_dir: str = "models"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.learning_rate <= 0 or self.learning_rate > 1:
            raise ValueError("Learning rate must be between 0 and 1")
        if self.discount_factor <= 0 or self.discount_factor > 1:
            raise ValueError("Discount factor must be between 0 and 1")
        if self.episodes <= 0:
            raise ValueError("Number of episodes must be positive")


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    # Plot settings
    style: str = "seaborn-v0_8"
    color_palette: str = "viridis"
    figure_size: tuple = (10, 8)
    dpi: int = 300
    
    # Q-table visualization
    q_table_cmap: str = "RdYlBu_r"
    show_values: bool = True
    
    # Policy visualization
    policy_symbols: list = field(default_factory=lambda: ['←', '↓', '→', '↑'])
    policy_cmap: str = "Set3"
    
    # Value function visualization
    value_cmap: str = "viridis"
    
    # Training progress
    smoothing_window: int = 50


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters."""
    # Metrics to compute
    compute_policy_consistency: bool = True
    compute_value_convergence: bool = True
    compute_action_distribution: bool = True
    compute_trajectory_analysis: bool = True
    
    # Consistency analysis
    consistency_window: int = 100
    min_episodes_for_consistency: int = 10
    
    # Convergence analysis
    convergence_threshold: float = 0.01
    convergence_window: int = 50


def load_config(config_path: str) -> RLConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return RLConfig(**config_dict)


def save_config(config: RLConfig, config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        config_path: Path where to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = {
        field.name: getattr(config, field.name) 
        for field in config.__dataclass_fields__.values()
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def create_default_configs() -> None:
    """Create default configuration files."""
    # Main RL config
    rl_config = RLConfig()
    save_config(rl_config, "configs/rl_config.yaml")
    
    # Visualization config
    viz_config = VisualizationConfig()
    save_config(viz_config, "configs/visualization_config.yaml")
    
    # Evaluation config
    eval_config = EvaluationConfig()
    save_config(eval_config, "configs/evaluation_config.yaml")


if __name__ == "__main__":
    create_default_configs()
    print("Default configuration files created in configs/ directory")
