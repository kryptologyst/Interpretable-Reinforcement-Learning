"""
Test suite for Interpretable RL framework.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from interpretable_rl import (
    InterpretableQLearningAgent,
    RLVisualizer,
    RLInterpretabilityEvaluator,
    RLConfig,
    VisualizationConfig,
    EvaluationConfig
)


class TestRLConfig:
    """Test RL configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = RLConfig()
        assert config.learning_rate == 0.1
        assert config.discount_factor == 0.99
        assert config.episodes == 1000
        assert config.epsilon == 0.1
        assert config.random_seed == 42
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid learning rate
        with pytest.raises(ValueError):
            RLConfig(learning_rate=1.5)
        
        # Test invalid discount factor
        with pytest.raises(ValueError):
            RLConfig(discount_factor=1.5)
        
        # Test invalid episodes
        with pytest.raises(ValueError):
            RLConfig(episodes=-1)


class TestInterpretableQLearningAgent:
    """Test Q-learning agent functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RLConfig(episodes=10, random_seed=42)
    
    @pytest.fixture
    def agent(self, config):
        """Create test agent."""
        agent = InterpretableQLearningAgent(config)
        agent.create_environment()
        agent.initialize_q_table()
        return agent
    
    def test_agent_initialization(self, config):
        """Test agent initialization."""
        agent = InterpretableQLearningAgent(config)
        assert agent.config == config
        assert agent.env is None
        assert agent.q_table is None
        assert len(agent.training_history['episode_rewards']) == 0
    
    def test_environment_creation(self, agent):
        """Test environment creation."""
        assert agent.env is not None
        assert hasattr(agent.env, 'observation_space')
        assert hasattr(agent.env, 'action_space')
    
    def test_q_table_initialization(self, agent):
        """Test Q-table initialization."""
        assert agent.q_table is not None
        assert agent.q_table.shape == (16, 4)  # FrozenLake 4x4 grid
        assert np.all(agent.q_table == 0)
    
    def test_action_selection(self, agent):
        """Test action selection."""
        state = 0
        epsilon = 0.0  # Greedy
        
        # Should select best action (initially random since all Q-values are 0)
        action = agent.get_action(state, epsilon)
        assert 0 <= action <= 3
    
    def test_training(self, agent):
        """Test agent training."""
        q_table, history = agent.train()
        
        # Check training completed
        assert len(history['episode_rewards']) == agent.config.episodes
        assert len(history['episode_lengths']) == agent.config.episodes
        assert len(history['epsilon_values']) == agent.config.episodes
        
        # Check Q-table was updated
        assert not np.all(q_table == 0)
    
    def test_policy_extraction(self, agent):
        """Test policy extraction."""
        agent.train()
        policy = agent.get_policy()
        
        assert len(policy) == 16
        assert all(0 <= action <= 3 for action in policy)
    
    def test_value_function_extraction(self, agent):
        """Test value function extraction."""
        agent.train()
        value_function = agent.get_value_function()
        
        assert len(value_function) == 16
        assert all(v >= 0 for v in value_function)
    
    def test_evaluation(self, agent):
        """Test agent evaluation."""
        agent.train()
        eval_results = agent.evaluate(num_episodes=5)
        
        assert 'mean_reward' in eval_results
        assert 'success_rate' in eval_results
        assert 'mean_length' in eval_results
        assert 0 <= eval_results['success_rate'] <= 1
    
    def test_model_save_load(self, agent, tmp_path):
        """Test model saving and loading."""
        agent.train()
        model_path = tmp_path / "test_model.npz"
        
        # Save model
        agent.save_model(str(model_path))
        assert model_path.exists()
        
        # Create new agent and load model
        new_agent = InterpretableQLearningAgent(agent.config)
        new_agent.load_model(str(model_path))
        
        # Check Q-table was loaded correctly
        assert np.allclose(agent.q_table, new_agent.q_table)


class TestRLVisualizer:
    """Test visualization functionality."""
    
    @pytest.fixture
    def trained_agent(self):
        """Create trained agent for visualization tests."""
        config = RLConfig(episodes=50, random_seed=42)
        agent = InterpretableQLearningAgent(config)
        agent.create_environment()
        agent.initialize_q_table()
        agent.train()
        return agent
    
    @pytest.fixture
    def visualizer(self, trained_agent):
        """Create visualizer with trained agent."""
        viz_config = VisualizationConfig()
        return RLVisualizer(trained_agent, viz_config)
    
    def test_q_table_plot(self, visualizer):
        """Test Q-table plotting."""
        fig = visualizer.plot_q_table()
        assert fig is not None
        assert len(fig.axes) == 1
    
    def test_policy_plot(self, visualizer):
        """Test policy plotting."""
        fig = visualizer.plot_policy()
        assert fig is not None
        assert len(fig.axes) == 1
    
    def test_value_function_plot(self, visualizer):
        """Test value function plotting."""
        fig = visualizer.plot_value_function()
        assert fig is not None
        assert len(fig.axes) == 1
    
    def test_training_progress_plot(self, visualizer):
        """Test training progress plotting."""
        fig = visualizer.plot_training_progress()
        assert fig is not None
        assert len(fig.axes) == 4  # 2x2 subplot
    
    def test_action_distribution_plot(self, visualizer):
        """Test action distribution plotting."""
        fig = visualizer.plot_action_distribution()
        assert fig is not None
        assert len(fig.axes) == 1
    
    def test_interactive_q_table(self, visualizer):
        """Test interactive Q-table creation."""
        fig = visualizer.plot_interactive_q_table()
        assert fig is not None
        assert hasattr(fig, 'data')


class TestRLInterpretabilityEvaluator:
    """Test interpretability evaluation functionality."""
    
    @pytest.fixture
    def trained_agent(self):
        """Create trained agent for evaluation tests."""
        config = RLConfig(episodes=100, random_seed=42)
        agent = InterpretableQLearningAgent(config)
        agent.create_environment()
        agent.initialize_q_table()
        agent.train()
        return agent
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator."""
        eval_config = EvaluationConfig()
        return RLInterpretabilityEvaluator(eval_config)
    
    def test_policy_consistency_evaluation(self, evaluator, trained_agent):
        """Test policy consistency evaluation."""
        metrics = evaluator.evaluate_policy_consistency(trained_agent)
        
        if "error" not in metrics:
            assert "average_consistency" in metrics
            assert "states_analyzed" in metrics
            assert 0 <= metrics["average_consistency"] <= 1
    
    def test_value_convergence_evaluation(self, evaluator, trained_agent):
        """Test value convergence evaluation."""
        metrics = evaluator.evaluate_value_convergence(trained_agent)
        
        if "error" not in metrics:
            assert "convergence_score" in metrics
            assert "stability_score" in metrics
            assert 0 <= metrics["convergence_score"] <= 1
    
    def test_action_distribution_evaluation(self, evaluator, trained_agent):
        """Test action distribution evaluation."""
        metrics = evaluator.evaluate_action_distribution(trained_agent)
        
        if "error" not in metrics:
            assert "action_entropy" in metrics
            assert "action_diversity" in metrics
            assert 0 <= metrics["action_diversity"] <= 1
    
    def test_trajectory_efficiency_evaluation(self, evaluator, trained_agent):
        """Test trajectory efficiency evaluation."""
        metrics = evaluator.evaluate_trajectory_efficiency(trained_agent)
        
        if "error" not in metrics:
            assert "mean_efficiency" in metrics
            assert "success_rate" in metrics
            assert 0 <= metrics["success_rate"] <= 1
    
    def test_comprehensive_evaluation(self, evaluator, trained_agent):
        """Test comprehensive evaluation."""
        metrics = evaluator.evaluate_comprehensive(trained_agent)
        
        assert hasattr(metrics, 'overall_score')
        assert hasattr(metrics, 'policy_consistency')
        assert hasattr(metrics, 'value_convergence_score')
        assert 0 <= metrics.overall_score <= 1
    
    def test_evaluation_report(self, evaluator, trained_agent):
        """Test evaluation report generation."""
        report = evaluator.generate_evaluation_report(trained_agent)
        
        assert "overall_metrics" in report
        assert "detailed_metrics" in report
        assert "training_summary" in report
        assert "overall_score" in report["overall_metrics"]


class TestIntegration:
    """Integration tests for the full framework."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from training to evaluation."""
        # Configuration
        config = RLConfig(episodes=50, random_seed=42)
        viz_config = VisualizationConfig()
        eval_config = EvaluationConfig()
        
        # Train agent
        agent = InterpretableQLearningAgent(config)
        agent.create_environment()
        agent.initialize_q_table()
        q_table, history = agent.train()
        
        # Evaluate performance
        eval_results = agent.evaluate(num_episodes=10)
        assert 'success_rate' in eval_results
        
        # Create visualizations
        visualizer = RLVisualizer(agent, viz_config)
        fig = visualizer.plot_q_table()
        assert fig is not None
        
        # Run interpretability evaluation
        evaluator = RLInterpretabilityEvaluator(eval_config)
        metrics = evaluator.evaluate_comprehensive(agent)
        assert metrics.overall_score >= 0
        
        # Generate report
        report = evaluator.generate_evaluation_report(agent)
        assert "overall_metrics" in report


if __name__ == "__main__":
    pytest.main([__file__])
