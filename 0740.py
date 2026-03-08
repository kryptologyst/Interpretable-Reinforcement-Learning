#!/usr/bin/env python3
"""
Interpretable Reinforcement Learning - Main Script

This script demonstrates the modernized interpretable RL framework with
comprehensive visualization, evaluation, and analysis capabilities.

DISCLAIMER: This project is for research and educational purposes only.
XAI outputs may be unstable or misleading and should not be used for
regulated decisions without human review.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any

from interpretable_rl import (
    InterpretableQLearningAgent,
    RLVisualizer,
    RLInterpretabilityEvaluator,
    RLConfig,
    VisualizationConfig,
    EvaluationConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate interpretable RL."""
    print("=" * 80)
    print("INTERPRETABLE REINFORCEMENT LEARNING DEMONSTRATION")
    print("=" * 80)
    print("DISCLAIMER: This project is for research and educational purposes only.")
    print("XAI outputs may be unstable or misleading and should not be used for")
    print("regulated decisions without human review.")
    print("=" * 80)
    
    # Configuration
    rl_config = RLConfig(
        learning_rate=0.1,
        discount_factor=0.99,
        episodes=1000,
        epsilon=0.1,
        random_seed=42
    )
    
    viz_config = VisualizationConfig()
    eval_config = EvaluationConfig()
    
    logger.info("Starting interpretable RL demonstration")
    
    # Create and train agent
    logger.info("Creating and training Q-learning agent...")
    agent = InterpretableQLearningAgent(rl_config)
    agent.create_environment()
    agent.initialize_q_table()
    q_table, history = agent.train()
    
    # Evaluate agent performance
    logger.info("Evaluating agent performance...")
    eval_results = agent.evaluate(num_episodes=100)
    print(f"\nAgent Performance:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
    print(f"  Success Rate: {eval_results['success_rate']:.3f}")
    print(f"  Mean Episode Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
    
    # Create visualizer
    logger.info("Creating visualizations...")
    visualizer = RLVisualizer(agent, viz_config)
    
    # Generate and save visualizations
    print("\nGenerating interpretability visualizations...")
    visualizer.save_all_plots(save_dir="assets")
    
    # Create evaluator and run comprehensive evaluation
    logger.info("Running interpretability evaluation...")
    evaluator = RLInterpretabilityEvaluator(eval_config)
    interpretability_report = evaluator.generate_evaluation_report(agent)
    
    # Display interpretability metrics
    print(f"\nInterpretability Analysis:")
    print(f"  Overall Score: {interpretability_report['overall_metrics']['overall_score']:.3f}")
    print(f"  Policy Consistency: {interpretability_report['overall_metrics']['policy_consistency']:.3f}")
    print(f"  Value Convergence: {interpretability_report['overall_metrics']['value_convergence_score']:.3f}")
    print(f"  Value Stability: {interpretability_report['overall_metrics']['value_stability']:.3f}")
    print(f"  Action Diversity: {interpretability_report['overall_metrics']['action_diversity']:.3f}")
    print(f"  Trajectory Efficiency: {interpretability_report['overall_metrics']['trajectory_efficiency']:.3f}")
    
    # Policy analysis
    policy_consistency = agent.analyze_policy_consistency()
    if "error" not in policy_consistency:
        print(f"\nPolicy Consistency Analysis:")
        print(f"  Average Consistency: {policy_consistency['average_consistency']:.3f}")
        print(f"  States Analyzed: {policy_consistency['states_analyzed']}")
        print(f"  Consistency Range: {policy_consistency['min_consistency']:.3f} - {policy_consistency['max_consistency']:.3f}")
    
    # Training summary
    training_summary = agent.get_training_summary()
    print(f"\nTraining Summary:")
    print(f"  Total Episodes: {training_summary['total_episodes']}")
    print(f"  Final Success Rate: {training_summary['final_success_rate']:.3f}")
    print(f"  Final Epsilon: {training_summary['final_epsilon']:.3f}")
    
    # Save model and report
    logger.info("Saving model and evaluation report...")
    agent.save_model("models/trained_agent.npz")
    
    import json
    with open("assets/evaluation_report.json", "w") as f:
        json.dump(interpretability_report, f, indent=2, default=str)
    
    print(f"\nTraining completed successfully!")
    print(f"Visualizations saved to assets/ directory")
    print(f"Model saved to models/trained_agent.npz")
    print(f"Evaluation report saved to assets/evaluation_report.json")
    
    # Display final Q-table and policy
    print(f"\nFinal Q-table shape: {q_table.shape}")
    print(f"Final policy (first 8 states): {agent.get_policy()[:8]}")
    print(f"Final value function (first 8 states): {agent.get_value_function()[:8]}")


if __name__ == "__main__":
    main()

