"""
Evaluation framework for RL interpretability metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from .agents.q_learning_agent import InterpretableQLearningAgent
from .config import EvaluationConfig

logger = logging.getLogger(__name__)


@dataclass
class InterpretabilityMetrics:
    """Container for interpretability evaluation metrics."""
    
    # Policy consistency metrics
    policy_consistency: float = 0.0
    min_consistency: float = 0.0
    max_consistency: float = 0.0
    states_analyzed: int = 0
    
    # Value function convergence metrics
    value_convergence_score: float = 0.0
    value_stability: float = 0.0
    
    # Action distribution metrics
    action_entropy: float = 0.0
    action_diversity: float = 0.0
    
    # Trajectory analysis metrics
    trajectory_efficiency: float = 0.0
    trajectory_consistency: float = 0.0
    
    # Overall interpretability score
    overall_score: float = 0.0


class RLInterpretabilityEvaluator:
    """
    Evaluator for RL interpretability metrics.
    
    This class provides comprehensive evaluation of interpretability aspects
    of reinforcement learning agents including policy consistency, value
    function convergence, and trajectory analysis.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize the evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
    
    def evaluate_policy_consistency(self, agent: InterpretableQLearningAgent) -> Dict[str, float]:
        """
        Evaluate policy consistency across training episodes.
        
        Args:
            agent: Trained Q-learning agent
            
        Returns:
            Dictionary with consistency metrics
        """
        if len(agent.trajectories) < self.config.min_episodes_for_consistency:
            logger.warning("Insufficient trajectories for consistency analysis")
            return {"error": "Insufficient trajectories for analysis"}
        
        # Analyze action consistency in similar states
        state_action_counts = {}
        recent_trajectories = agent.trajectories[-self.config.consistency_window:]
        
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
        
        if not consistency_scores:
            return {
                "average_consistency": 0.0,
                "min_consistency": 0.0,
                "max_consistency": 0.0,
                "states_analyzed": 0,
                "total_states": len(state_action_counts)
            }
        
        return {
            "average_consistency": np.mean(consistency_scores),
            "min_consistency": np.min(consistency_scores),
            "max_consistency": np.max(consistency_scores),
            "states_analyzed": len(consistency_scores),
            "total_states": len(state_action_counts),
            "consistency_std": np.std(consistency_scores)
        }
    
    def evaluate_value_convergence(self, agent: InterpretableQLearningAgent) -> Dict[str, float]:
        """
        Evaluate value function convergence and stability.
        
        Args:
            agent: Trained Q-learning agent
            
        Returns:
            Dictionary with convergence metrics
        """
        if not agent.training_history['q_table_changes']:
            return {"error": "No Q-table change data available"}
        
        q_changes = agent.training_history['q_table_changes']
        
        # Calculate convergence metrics
        recent_changes = q_changes[-self.config.convergence_window:]
        convergence_score = 1.0 - np.mean(recent_changes)  # Higher is better
        
        # Stability metric (lower variance in recent changes is better)
        stability_score = 1.0 / (1.0 + np.std(recent_changes))
        
        # Check if converged (changes below threshold)
        is_converged = np.mean(recent_changes) < self.config.convergence_threshold
        
        return {
            "convergence_score": convergence_score,
            "stability_score": stability_score,
            "is_converged": is_converged,
            "mean_recent_change": np.mean(recent_changes),
            "std_recent_change": np.std(recent_changes),
            "convergence_rate": self._calculate_convergence_rate(q_changes)
        }
    
    def _calculate_convergence_rate(self, q_changes: List[float]) -> float:
        """
        Calculate the rate of convergence based on Q-table changes.
        
        Args:
            q_changes: List of Q-table changes over episodes
            
        Returns:
            Convergence rate (0-1, higher is better)
        """
        if len(q_changes) < 2:
            return 0.0
        
        # Calculate the rate of decrease in changes
        changes_array = np.array(q_changes)
        if np.all(changes_array == 0):
            return 1.0
        
        # Fit exponential decay to changes
        try:
            from scipy.optimize import curve_fit
            
            def exponential_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            x = np.arange(len(changes_array))
            popt, _ = curve_fit(exponential_decay, x, changes_array, 
                              maxfev=1000, bounds=(0, [np.inf, 1, np.inf]))
            
            # Extract decay rate
            decay_rate = popt[1]
            convergence_rate = min(1.0, decay_rate * 10)  # Scale to 0-1
            
            return convergence_rate
        except:
            # Fallback: simple ratio of early vs late changes
            early_changes = np.mean(changes_array[:len(changes_array)//2])
            late_changes = np.mean(changes_array[len(changes_array)//2:])
            
            if early_changes == 0:
                return 1.0
            
            return min(1.0, late_changes / early_changes)
    
    def evaluate_action_distribution(self, agent: InterpretableQLearningAgent) -> Dict[str, float]:
        """
        Evaluate action distribution characteristics.
        
        Args:
            agent: Trained Q-learning agent
            
        Returns:
            Dictionary with action distribution metrics
        """
        if not agent.trajectories:
            return {"error": "No trajectory data available"}
        
        # Count actions in recent trajectories
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Left, Down, Right, Up
        recent_trajectories = agent.trajectories[-self.config.consistency_window:]
        
        for trajectory in recent_trajectories:
            for _, action, _, _ in trajectory:
                action_counts[action] += 1
        
        total_actions = sum(action_counts.values())
        if total_actions == 0:
            return {"error": "No actions found in trajectories"}
        
        # Calculate entropy (diversity measure)
        probabilities = [count / total_actions for count in action_counts.values()]
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
        max_entropy = np.log2(len(action_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Calculate diversity (1 - max probability)
        max_prob = max(probabilities)
        diversity = 1.0 - max_prob
        
        return {
            "action_entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "action_diversity": diversity,
            "max_action_probability": max_prob,
            "action_distribution": action_counts,
            "total_actions": total_actions
        }
    
    def evaluate_trajectory_efficiency(self, agent: InterpretableQLearningAgent) -> Dict[str, float]:
        """
        Evaluate trajectory efficiency and consistency.
        
        Args:
            agent: Trained Q-learning agent
            
        Returns:
            Dictionary with trajectory metrics
        """
        if not agent.trajectories:
            return {"error": "No trajectory data available"}
        
        recent_trajectories = agent.trajectories[-self.config.consistency_window:]
        
        # Calculate efficiency metrics
        episode_lengths = [len(trajectory) for trajectory in recent_trajectories]
        episode_rewards = [sum(reward for _, _, reward, _ in trajectory) 
                          for trajectory in recent_trajectories]
        
        # Efficiency: reward per step
        efficiency_scores = []
        for length, reward in zip(episode_lengths, episode_rewards):
            if length > 0:
                efficiency_scores.append(reward / length)
            else:
                efficiency_scores.append(0)
        
        # Consistency: low variance in trajectory lengths
        length_consistency = 1.0 / (1.0 + np.std(episode_lengths))
        
        # Success rate
        success_rate = sum(1 for reward in episode_rewards if reward > 0) / len(episode_rewards)
        
        return {
            "mean_efficiency": np.mean(efficiency_scores),
            "std_efficiency": np.std(efficiency_scores),
            "length_consistency": length_consistency,
            "success_rate": success_rate,
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards)
        }
    
    def evaluate_comprehensive(self, agent: InterpretableQLearningAgent) -> InterpretabilityMetrics:
        """
        Perform comprehensive interpretability evaluation.
        
        Args:
            agent: Trained Q-learning agent
            
        Returns:
            Comprehensive interpretability metrics
        """
        logger.info("Starting comprehensive interpretability evaluation")
        
        # Evaluate all aspects
        policy_metrics = self.evaluate_policy_consistency(agent)
        value_metrics = self.evaluate_value_convergence(agent)
        action_metrics = self.evaluate_action_distribution(agent)
        trajectory_metrics = self.evaluate_trajectory_efficiency(agent)
        
        # Extract key metrics
        metrics = InterpretabilityMetrics()
        
        # Policy consistency
        if "error" not in policy_metrics:
            metrics.policy_consistency = policy_metrics.get("average_consistency", 0.0)
            metrics.min_consistency = policy_metrics.get("min_consistency", 0.0)
            metrics.max_consistency = policy_metrics.get("max_consistency", 0.0)
            metrics.states_analyzed = policy_metrics.get("states_analyzed", 0)
        
        # Value function convergence
        if "error" not in value_metrics:
            metrics.value_convergence_score = value_metrics.get("convergence_score", 0.0)
            metrics.value_stability = value_metrics.get("stability_score", 0.0)
        
        # Action distribution
        if "error" not in action_metrics:
            metrics.action_entropy = action_metrics.get("normalized_entropy", 0.0)
            metrics.action_diversity = action_metrics.get("action_diversity", 0.0)
        
        # Trajectory analysis
        if "error" not in trajectory_metrics:
            metrics.trajectory_efficiency = trajectory_metrics.get("mean_efficiency", 0.0)
            metrics.trajectory_consistency = trajectory_metrics.get("length_consistency", 0.0)
        
        # Calculate overall interpretability score
        metrics.overall_score = self._calculate_overall_score(metrics)
        
        logger.info(f"Interpretability evaluation completed. Overall score: {metrics.overall_score:.3f}")
        
        return metrics
    
    def _calculate_overall_score(self, metrics: InterpretabilityMetrics) -> float:
        """
        Calculate overall interpretability score.
        
        Args:
            metrics: Individual interpretability metrics
            
        Returns:
            Overall interpretability score (0-1)
        """
        # Weighted combination of different aspects
        weights = {
            'policy_consistency': 0.3,
            'value_convergence': 0.25,
            'value_stability': 0.15,
            'action_diversity': 0.15,
            'trajectory_efficiency': 0.15
        }
        
        score = (
            weights['policy_consistency'] * metrics.policy_consistency +
            weights['value_convergence'] * metrics.value_convergence_score +
            weights['value_stability'] * metrics.value_stability +
            weights['action_diversity'] * metrics.action_diversity +
            weights['trajectory_efficiency'] * metrics.trajectory_efficiency
        )
        
        return min(1.0, max(0.0, score))
    
    def generate_evaluation_report(self, agent: InterpretableQLearningAgent) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            agent: Trained Q-learning agent
            
        Returns:
            Comprehensive evaluation report
        """
        metrics = self.evaluate_comprehensive(agent)
        
        # Get detailed metrics
        policy_metrics = self.evaluate_policy_consistency(agent)
        value_metrics = self.evaluate_value_convergence(agent)
        action_metrics = self.evaluate_action_distribution(agent)
        trajectory_metrics = self.evaluate_trajectory_efficiency(agent)
        
        # Training summary
        training_summary = agent.get_training_summary()
        
        report = {
            "overall_metrics": {
                "overall_score": metrics.overall_score,
                "policy_consistency": metrics.policy_consistency,
                "value_convergence_score": metrics.value_convergence_score,
                "value_stability": metrics.value_stability,
                "action_entropy": metrics.action_entropy,
                "action_diversity": metrics.action_diversity,
                "trajectory_efficiency": metrics.trajectory_efficiency,
                "trajectory_consistency": metrics.trajectory_consistency
            },
            "detailed_metrics": {
                "policy_consistency": policy_metrics,
                "value_convergence": value_metrics,
                "action_distribution": action_metrics,
                "trajectory_efficiency": trajectory_metrics
            },
            "training_summary": training_summary,
            "evaluation_config": {
                "consistency_window": self.config.consistency_window,
                "convergence_threshold": self.config.convergence_threshold,
                "convergence_window": self.config.convergence_window
            }
        }
        
        return report
