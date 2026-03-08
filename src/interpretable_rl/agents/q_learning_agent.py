"""
Interpretable Q-Learning Agent implementation.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import Env
from dataclasses import dataclass

from .config import RLConfig

logger = logging.getLogger(__name__)


class InterpretableQLearningAgent:
    """
    An interpretable Q-learning agent with comprehensive analysis capabilities.
    
    This class implements Q-learning with built-in interpretability features
    including Q-table visualization, policy analysis, and trajectory tracking.
    """
    
    def __init__(self, config: RLConfig):
        """
        Initialize the Q-learning agent.
        
        Args:
            config: Configuration object containing training parameters
        """
        self.config = config
        self.env: Optional[Env] = None
        self.q_table: Optional[np.ndarray] = None
        
        # Training history tracking
        self.training_history: Dict[str, List[float]] = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_values': [],
            'q_table_changes': [],
            'success_rate': []
        }
        
        # Trajectory storage for analysis
        self.trajectories: List[List[Tuple[int, int, float, int]]] = []
        
        # Set random seeds for reproducibility
        np.random.seed(config.random_seed)
        
    def create_environment(self) -> Env:
        """
        Create and configure the RL environment.
        
        Returns:
            Configured gymnasium environment
        """
        try:
            self.env = gym.make(
                self.config.environment_name,
                is_slippery=self.config.is_slippery,
                render_mode=None
            )
            logger.info(f"Created environment: {self.config.environment_name}")
            return self.env
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            raise
    
    def initialize_q_table(self) -> np.ndarray:
        """
        Initialize the Q-table with zeros.
        
        Returns:
            Initialized Q-table
        """
        if self.env is None:
            raise ValueError("Environment must be created first")
            
        state_space = self.env.observation_space.n
        action_space = self.env.action_space.n
        
        self.q_table = np.zeros((state_space, action_space))
        logger.info(f"Initialized Q-table with shape: {self.q_table.shape}")
        return self.q_table
    
    def get_action(self, state: int, epsilon: float) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate
            
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def train(self) -> Tuple[np.ndarray, Dict[str, List[float]]]:
        """
        Train the Q-learning agent.
        
        Returns:
            Tuple of (trained_q_table, training_history)
        """
        if self.env is None or self.q_table is None:
            raise ValueError("Environment and Q-table must be initialized first")
        
        logger.info(f"Starting training for {self.config.episodes} episodes")
        
        epsilon = self.config.epsilon
        total_q_changes = []
        success_count = 0
        
        for episode in range(self.config.episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            episode_length = 0
            episode_trajectory = []
            q_table_before = self.q_table.copy()
            
            while not done:
                action = self.get_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store trajectory for analysis
                episode_trajectory.append((state, action, reward, next_state))
                
                # Q-learning update
                old_q_value = self.q_table[state, action]
                best_next_action = np.argmax(self.q_table[next_state])
                td_target = reward + self.config.discount_factor * self.q_table[next_state, best_next_action]
                self.q_table[state, action] = (1 - self.config.learning_rate) * old_q_value + \
                                            self.config.learning_rate * td_target
                
                total_reward += reward
                episode_length += 1
                state = next_state
            
            # Track success (reward > 0 indicates reaching goal)
            if total_reward > 0:
                success_count += 1
            
            # Store episode data
            self.training_history['episode_rewards'].append(total_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['epsilon_values'].append(epsilon)
            self.training_history['success_rate'].append(success_count / (episode + 1))
            
            # Track Q-table changes for stability analysis
            q_change = np.mean(np.abs(self.q_table - q_table_before))
            total_q_changes.append(q_change)
            
            # Store trajectory
            self.trajectories.append(episode_trajectory)
            
            # Decay epsilon
            epsilon = max(self.config.epsilon_min, epsilon * self.config.epsilon_decay)
            
            if episode % self.config.eval_frequency == 0:
                logger.info(f"Episode {episode}: Reward = {total_reward:.2f}, "
                          f"Epsilon = {epsilon:.3f}, Length = {episode_length}, "
                          f"Success Rate = {success_count / (episode + 1):.3f}")
        
        self.training_history['q_table_changes'] = total_q_changes
        logger.info("Training completed")
        
        return self.q_table, self.training_history
    
    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.env is None or self.q_table is None:
            raise ValueError("Agent must be trained first")
        
        total_rewards = []
        episode_lengths = []
        success_count = 0
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            episode_length = 0
            
            while not done:
                action = np.argmax(self.q_table[state])  # Greedy policy
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                episode_length += 1
                state = next_state
            
            total_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            
            if total_reward > 0:
                success_count += 1
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': success_count / num_episodes,
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards)
        }
    
    def get_policy(self) -> np.ndarray:
        """
        Extract the greedy policy from the Q-table.
        
        Returns:
            Policy array where each state maps to its best action
        """
        if self.q_table is None:
            raise ValueError("Q-table not initialized")
        return np.argmax(self.q_table, axis=1)
    
    def get_value_function(self) -> np.ndarray:
        """
        Extract the value function from the Q-table.
        
        Returns:
            Value function (max Q-value for each state)
        """
        if self.q_table is None:
            raise ValueError("Q-table not initialized")
        return np.max(self.q_table, axis=1)
    
    def get_action_values(self, state: int) -> np.ndarray:
        """
        Get Q-values for all actions in a given state.
        
        Args:
            state: State index
            
        Returns:
            Array of Q-values for all actions
        """
        if self.q_table is None:
            raise ValueError("Q-table not initialized")
        return self.q_table[state, :]
    
    def analyze_policy_consistency(self) -> Dict[str, float]:
        """
        Analyze policy consistency across training.
        
        Returns:
            Dictionary with consistency metrics
        """
        if len(self.trajectories) < self.config.eval_episodes:
            return {"error": "Insufficient trajectories for analysis"}
        
        # Analyze action consistency in similar states
        state_action_counts = {}
        recent_trajectories = self.trajectories[-self.config.eval_episodes:]
        
        for trajectory in recent_trajectories:
            for state, action, _, _ in trajectory:
                if state not in state_action_counts:
                    state_action_counts[state] = {}
                state_action_counts[state][action] = state_action_counts[state].get(action, 0) + 1
        
        # Calculate consistency metrics
        consistency_scores = []
        for state, actions in state_action_counts.items():
            if len(actions) > 1:
                total_actions = sum(actions.values())
                max_action_count = max(actions.values())
                consistency = max_action_count / total_actions
                consistency_scores.append(consistency)
        
        return {
            "average_consistency": np.mean(consistency_scores) if consistency_scores else 0,
            "min_consistency": np.min(consistency_scores) if consistency_scores else 0,
            "max_consistency": np.max(consistency_scores) if consistency_scores else 0,
            "states_analyzed": len(consistency_scores),
            "total_states": len(state_action_counts)
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of training results.
        
        Returns:
            Dictionary with training summary
        """
        if not self.training_history['episode_rewards']:
            return {"error": "No training data available"}
        
        recent_rewards = self.training_history['episode_rewards'][-100:]
        recent_lengths = self.training_history['episode_lengths'][-100:]
        
        return {
            "total_episodes": len(self.training_history['episode_rewards']),
            "final_success_rate": self.training_history['success_rate'][-1],
            "mean_recent_reward": np.mean(recent_rewards),
            "std_recent_reward": np.std(recent_rewards),
            "mean_recent_length": np.mean(recent_lengths),
            "std_recent_length": np.std(recent_lengths),
            "final_epsilon": self.training_history['epsilon_values'][-1],
            "q_table_shape": self.q_table.shape if self.q_table is not None else None
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.q_table is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'q_table': self.q_table,
            'config': self.config,
            'training_history': self.training_history
        }
        
        np.savez(filepath, **model_data)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to load the model from
        """
        data = np.load(filepath, allow_pickle=True)
        self.q_table = data['q_table']
        self.training_history = data['training_history'].item()
        logger.info(f"Model loaded from {filepath}")
