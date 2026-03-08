"""
Comprehensive visualization tools for RL interpretability.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .agents.q_learning_agent import InterpretableQLearningAgent
from .config import VisualizationConfig


class RLVisualizer:
    """Comprehensive visualization tools for RL interpretability."""
    
    def __init__(self, agent: InterpretableQLearningAgent, config: VisualizationConfig):
        """
        Initialize visualizer with trained agent.
        
        Args:
            agent: Trained Q-learning agent
            config: Visualization configuration
        """
        self.agent = agent
        self.config = config
        self.env = agent.env
        
        # Set matplotlib style
        plt.style.use(config.style)
        
    def plot_q_table(self, figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Visualize the Q-table as a heatmap.
        
        Args:
            figsize: Figure size (uses config default if None)
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = self.config.figure_size
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(self.agent.q_table, cmap=self.config.q_table_cmap, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Q-Value', rotation=270, labelpad=20)
        
        # Set labels
        ax.set_xlabel('Action')
        ax.set_ylabel('State')
        ax.set_title('Q-Table Heatmap')
        
        # Set action labels
        action_labels = ['Left', 'Down', 'Right', 'Up']
        ax.set_xticks(range(len(action_labels)))
        ax.set_xticklabels(action_labels)
        
        # Add value annotations if enabled
        if self.config.show_values:
            for i in range(self.agent.q_table.shape[0]):
                for j in range(self.agent.q_table.shape[1]):
                    text = ax.text(j, i, f'{self.agent.q_table[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_policy(self, figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Visualize the learned policy.
        
        Args:
            figsize: Figure size (uses config default if None)
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = self.config.figure_size
            
        fig, ax = plt.subplots(figsize=figsize)
        
        policy = self.agent.get_policy()
        
        # Create policy visualization
        policy_symbols = [self.config.policy_symbols[action] for action in policy]
        
        # Reshape for grid visualization (assuming square grid)
        grid_size = int(np.sqrt(len(policy)))
        policy_grid = np.array(policy_symbols).reshape(grid_size, grid_size)
        
        # Create heatmap with policy symbols
        im = ax.imshow(policy.reshape(grid_size, grid_size), 
                      cmap=self.config.policy_cmap, aspect='equal')
        
        # Add policy symbols
        for i in range(grid_size):
            for j in range(grid_size):
                ax.text(j, i, policy_grid[i, j], ha="center", va="center", 
                       fontsize=20, fontweight='bold')
        
        ax.set_title('Learned Policy')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        plt.tight_layout()
        return fig
    
    def plot_value_function(self, figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Visualize the value function.
        
        Args:
            figsize: Figure size (uses config default if None)
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = self.config.figure_size
            
        fig, ax = plt.subplots(figsize=figsize)
        
        value_function = self.agent.get_value_function()
        
        # Reshape for grid visualization
        grid_size = int(np.sqrt(len(value_function)))
        value_grid = value_function.reshape(grid_size, grid_size)
        
        # Create heatmap
        im = ax.imshow(value_grid, cmap=self.config.value_cmap, aspect='equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('State Value', rotation=270, labelpad=20)
        
        # Add value annotations
        for i in range(grid_size):
            for j in range(grid_size):
                text = ax.text(j, i, f'{value_grid[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=10)
        
        ax.set_title('Value Function')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        plt.tight_layout()
        return fig
    
    def plot_training_progress(self, figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot training progress metrics.
        
        Args:
            figsize: Figure size (uses config default if None)
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = (15, 10)
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        history = self.agent.training_history
        
        # Episode rewards with smoothing
        rewards = np.array(history['episode_rewards'])
        if len(rewards) > self.config.smoothing_window:
            smoothed_rewards = np.convolve(rewards, 
                                         np.ones(self.config.smoothing_window)/self.config.smoothing_window, 
                                         mode='valid')
            axes[0, 0].plot(rewards, alpha=0.3, label='Raw')
            axes[0, 0].plot(range(self.config.smoothing_window-1, len(rewards)), 
                          smoothed_rewards, label='Smoothed')
        else:
            axes[0, 0].plot(rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Episode lengths
        axes[0, 1].plot(history['episode_lengths'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        # Epsilon decay
        axes[1, 0].plot(history['epsilon_values'])
        axes[1, 0].set_title('Epsilon Decay')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True)
        
        # Success rate
        axes[1, 1].plot(history['success_rate'])
        axes[1, 1].set_title('Success Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_action_distribution(self, figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot action distribution across states.
        
        Args:
            figsize: Figure size (uses config default if None)
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = self.config.figure_size
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Count actions in trajectories
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Left, Down, Right, Up
        
        recent_trajectories = self.agent.trajectories[-100:]  # Last 100 episodes
        for trajectory in recent_trajectories:
            for _, action, _, _ in trajectory:
                action_counts[action] += 1
        
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        action_labels = ['Left', 'Down', 'Right', 'Up']
        
        bars = ax.bar(action_labels, counts, color=['red', 'blue', 'green', 'orange'])
        ax.set_title('Action Distribution in Recent Episodes')
        ax.set_xlabel('Action')
        ax.set_ylabel('Count')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_state_action_values(self, state: int, figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot Q-values for all actions in a specific state.
        
        Args:
            state: State index to analyze
            figsize: Figure size (uses config default if None)
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = self.config.figure_size
            
        fig, ax = plt.subplots(figsize=figsize)
        
        action_values = self.agent.get_action_values(state)
        action_labels = ['Left', 'Down', 'Right', 'Up']
        
        bars = ax.bar(action_labels, action_values, color=['red', 'blue', 'green', 'orange'])
        ax.set_title(f'Q-Values for State {state}')
        ax.set_xlabel('Action')
        ax.set_ylabel('Q-Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, action_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_interactive_q_table(self) -> go.Figure:
        """
        Create an interactive Q-table visualization using Plotly.
        
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=self.agent.q_table,
            colorscale='RdYlBu_r',
            text=np.round(self.agent.q_table, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Interactive Q-Table',
            xaxis_title='Action',
            yaxis_title='State',
            xaxis=dict(
                tickmode='array',
                tickvals=[0, 1, 2, 3],
                ticktext=['Left', 'Down', 'Right', 'Up']
            )
        )
        
        return fig
    
    def plot_interactive_training_progress(self) -> go.Figure:
        """
        Create an interactive training progress visualization.
        
        Returns:
            Plotly figure with subplots
        """
        history = self.agent.training_history
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Episode Rewards', 'Episode Lengths', 
                          'Epsilon Decay', 'Success Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Episode rewards
        fig.add_trace(
            go.Scatter(y=history['episode_rewards'], mode='lines', name='Rewards'),
            row=1, col=1
        )
        
        # Episode lengths
        fig.add_trace(
            go.Scatter(y=history['episode_lengths'], mode='lines', name='Lengths'),
            row=1, col=2
        )
        
        # Epsilon decay
        fig.add_trace(
            go.Scatter(y=history['epsilon_values'], mode='lines', name='Epsilon'),
            row=2, col=1
        )
        
        # Success rate
        fig.add_trace(
            go.Scatter(y=history['success_rate'], mode='lines', name='Success Rate'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Training Progress Dashboard")
        
        return fig
    
    def save_all_plots(self, save_dir: str = "assets") -> None:
        """
        Save all visualization plots to files.
        
        Args:
            save_dir: Directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Q-table heatmap
        fig1 = self.plot_q_table()
        fig1.savefig(f'{save_dir}/q_table_heatmap.png', 
                    dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig1)
        
        # Policy visualization
        fig2 = self.plot_policy()
        fig2.savefig(f'{save_dir}/learned_policy.png', 
                    dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig2)
        
        # Value function
        fig3 = self.plot_value_function()
        fig3.savefig(f'{save_dir}/value_function.png', 
                    dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig3)
        
        # Training progress
        fig4 = self.plot_training_progress()
        fig4.savefig(f'{save_dir}/training_progress.png', 
                    dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig4)
        
        # Action distribution
        fig5 = self.plot_action_distribution()
        fig5.savefig(f'{save_dir}/action_distribution.png', 
                    dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig5)
        
        print(f"All plots saved to {save_dir}/ directory")
