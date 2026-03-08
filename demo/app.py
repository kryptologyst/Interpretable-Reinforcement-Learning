"""
Interactive Streamlit Demo for Interpretable Reinforcement Learning

This demo provides an interactive interface for exploring RL interpretability
with real-time visualization and analysis capabilities.

DISCLAIMER: This project is for research and educational purposes only.
XAI outputs may be unstable or misleading and should not be used for
regulated decisions without human review.
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from interpretable_rl import (
    InterpretableQLearningAgent,
    RLVisualizer,
    RLInterpretabilityEvaluator,
    RLConfig,
    VisualizationConfig,
    EvaluationConfig
)

# Page configuration
st.set_page_config(
    page_title="Interpretable RL Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Interpretable Reinforcement Learning Demo</h1>', 
                unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <h4>⚠️ DISCLAIMER</h4>
        <p>This project is for research and educational purposes only. 
        XAI outputs may be unstable or misleading and should not be used for 
        regulated decisions without human review.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    episodes = st.sidebar.slider("Episodes", 100, 2000, 1000, 100)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
    discount_factor = st.sidebar.slider("Discount Factor", 0.8, 0.99, 0.99, 0.01)
    epsilon = st.sidebar.slider("Initial Epsilon", 0.01, 0.5, 0.1, 0.01)
    random_seed = st.sidebar.number_input("Random Seed", 1, 1000, 42)
    
    # Environment parameters
    st.sidebar.subheader("Environment")
    is_slippery = st.sidebar.checkbox("Slippery Environment", False)
    
    # Visualization options
    st.sidebar.subheader("Visualization")
    show_values = st.sidebar.checkbox("Show Q-values on plots", True)
    plot_style = st.sidebar.selectbox("Plot Style", 
                                     ["seaborn-v0_8", "default", "ggplot", "dark_background"])
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Training", "📊 Analysis", "🎯 Interpretability", "📈 Metrics"])
    
    with tab1:
        st.header("Agent Training")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training agent..."):
                    # Create configuration
                    config = RLConfig(
                        learning_rate=learning_rate,
                        discount_factor=discount_factor,
                        episodes=episodes,
                        epsilon=epsilon,
                        random_seed=random_seed,
                        is_slippery=is_slippery
                    )
                    
                    # Create and train agent
                    agent = InterpretableQLearningAgent(config)
                    agent.create_environment()
                    agent.initialize_q_table()
                    q_table, history = agent.train()
                    
                    # Store in session state
                    st.session_state.agent = agent
                    st.session_state.training_complete = True
                    st.session_state.config = config
                    
                    st.success("Training completed successfully!")
        
        with col2:
            if st.session_state.get('training_complete', False):
                st.success("✅ Training Complete")
                agent = st.session_state.agent
                
                # Quick stats
                eval_results = agent.evaluate(num_episodes=50)
                st.metric("Success Rate", f"{eval_results['success_rate']:.1%}")
                st.metric("Mean Reward", f"{eval_results['mean_reward']:.2f}")
                st.metric("Mean Length", f"{eval_results['mean_length']:.1f}")
        
        # Training progress visualization
        if st.session_state.get('training_complete', False):
            st.subheader("Training Progress")
            
            agent = st.session_state.agent
            history = agent.training_history
            
            # Create training progress plot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Episode Rewards', 'Episode Lengths', 
                              'Epsilon Decay', 'Success Rate'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Episode rewards
            fig.add_trace(
                go.Scatter(y=history['episode_rewards'], mode='lines', 
                          name='Rewards', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Episode lengths
            fig.add_trace(
                go.Scatter(y=history['episode_lengths'], mode='lines', 
                          name='Lengths', line=dict(color='green')),
                row=1, col=2
            )
            
            # Epsilon decay
            fig.add_trace(
                go.Scatter(y=history['epsilon_values'], mode='lines', 
                          name='Epsilon', line=dict(color='orange')),
                row=2, col=1
            )
            
            # Success rate
            fig.add_trace(
                go.Scatter(y=history['success_rate'], mode='lines', 
                          name='Success Rate', line=dict(color='red')),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False, 
                             title_text="Training Progress Dashboard")
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Q-Table Analysis")
        
        if st.session_state.get('training_complete', False):
            agent = st.session_state.agent
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Q-Table Heatmap")
                
                # Interactive Q-table
                fig = go.Figure(data=go.Heatmap(
                    z=agent.q_table,
                    colorscale='RdYlBu_r',
                    text=np.round(agent.q_table, 3),
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
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Policy Visualization")
                
                policy = agent.get_policy()
                policy_symbols = ['←', '↓', '→', '↑']
                policy_display = [policy_symbols[action] for action in policy]
                
                # Reshape for grid display
                grid_size = int(np.sqrt(len(policy)))
                policy_grid = np.array(policy_display).reshape(grid_size, grid_size)
                
                # Create policy heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=policy.reshape(grid_size, grid_size),
                    colorscale='Set3',
                    text=policy_grid,
                    texttemplate="%{text}",
                    textfont={"size": 20},
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title='Learned Policy',
                    xaxis_title='Column',
                    yaxis_title='Row'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # State-specific analysis
            st.subheader("State-Specific Analysis")
            
            state_to_analyze = st.selectbox("Select State to Analyze", 
                                          range(len(agent.get_policy())))
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Q-values for selected state
                action_values = agent.get_action_values(state_to_analyze)
                action_labels = ['Left', 'Down', 'Right', 'Up']
                
                fig = px.bar(x=action_labels, y=action_values,
                           title=f'Q-Values for State {state_to_analyze}',
                           color=action_values,
                           color_continuous_scale='viridis')
                
                fig.update_layout(xaxis_title='Action', yaxis_title='Q-Value')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Value function visualization
                value_function = agent.get_value_function()
                grid_size = int(np.sqrt(len(value_function)))
                value_grid = value_function.reshape(grid_size, grid_size)
                
                fig = go.Figure(data=go.Heatmap(
                    z=value_grid,
                    colorscale='viridis',
                    text=np.round(value_grid, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title='Value Function',
                    xaxis_title='Column',
                    yaxis_title='Row'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Interpretability Analysis")
        
        if st.session_state.get('training_complete', False):
            agent = st.session_state.agent
            
            # Run interpretability evaluation
            eval_config = EvaluationConfig()
            evaluator = RLInterpretabilityEvaluator(eval_config)
            
            with st.spinner("Running interpretability analysis..."):
                interpretability_report = evaluator.generate_evaluation_report(agent)
            
            # Overall metrics
            st.subheader("Overall Interpretability Score")
            
            overall_score = interpretability_report['overall_metrics']['overall_score']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Score", f"{overall_score:.3f}")
            
            with col2:
                policy_consistency = interpretability_report['overall_metrics']['policy_consistency']
                st.metric("Policy Consistency", f"{policy_consistency:.3f}")
            
            with col3:
                value_convergence = interpretability_report['overall_metrics']['value_convergence_score']
                st.metric("Value Convergence", f"{value_convergence:.3f}")
            
            with col4:
                action_diversity = interpretability_report['overall_metrics']['action_diversity']
                st.metric("Action Diversity", f"{action_diversity:.3f}")
            
            # Detailed metrics
            st.subheader("Detailed Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Policy Consistency")
                policy_details = interpretability_report['detailed_metrics']['policy_consistency']
                if "error" not in policy_details:
                    st.write(f"**Average Consistency:** {policy_details['average_consistency']:.3f}")
                    st.write(f"**States Analyzed:** {policy_details['states_analyzed']}")
                    st.write(f"**Consistency Range:** {policy_details['min_consistency']:.3f} - {policy_details['max_consistency']:.3f}")
                else:
                    st.error("Insufficient data for policy consistency analysis")
            
            with col2:
                st.markdown("#### Value Function Convergence")
                value_details = interpretability_report['detailed_metrics']['value_convergence']
                if "error" not in value_details:
                    st.write(f"**Convergence Score:** {value_details['convergence_score']:.3f}")
                    st.write(f"**Stability Score:** {value_details['stability_score']:.3f}")
                    st.write(f"**Converged:** {'Yes' if value_details['is_converged'] else 'No'}")
                else:
                    st.error("No convergence data available")
            
            # Action distribution analysis
            st.subheader("Action Distribution Analysis")
            
            action_details = interpretability_report['detailed_metrics']['action_distribution']
            if "error" not in action_details:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Action Entropy:** {action_details['action_entropy']:.3f}")
                    st.write(f"**Action Diversity:** {action_details['action_diversity']:.3f}")
                    st.write(f"**Max Action Probability:** {action_details['max_action_probability']:.3f}")
                
                with col2:
                    # Action distribution pie chart
                    action_dist = action_details['action_distribution']
                    action_labels = ['Left', 'Down', 'Right', 'Up']
                    
                    fig = px.pie(values=list(action_dist.values()), 
                               names=action_labels,
                               title='Action Distribution')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No action distribution data available")
    
    with tab4:
        st.header("Performance Metrics")
        
        if st.session_state.get('training_complete', False):
            agent = st.session_state.agent
            
            # Evaluation metrics
            st.subheader("Agent Performance")
            
            eval_results = agent.evaluate(num_episodes=100)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Reward", f"{eval_results['mean_reward']:.3f}")
                st.metric("Reward Std", f"{eval_results['std_reward']:.3f}")
            
            with col2:
                st.metric("Success Rate", f"{eval_results['success_rate']:.1%}")
                st.metric("Min Reward", f"{eval_results['min_reward']:.3f}")
            
            with col3:
                st.metric("Mean Length", f"{eval_results['mean_length']:.1f}")
                st.metric("Length Std", f"{eval_results['std_length']:.1f}")
            
            with col4:
                st.metric("Max Reward", f"{eval_results['max_reward']:.3f}")
            
            # Training summary
            st.subheader("Training Summary")
            
            training_summary = agent.get_training_summary()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Total Episodes:** {training_summary['total_episodes']}")
                st.write(f"**Final Success Rate:** {training_summary['final_success_rate']:.3f}")
                st.write(f"**Final Epsilon:** {training_summary['final_epsilon']:.3f}")
            
            with col2:
                st.write(f"**Mean Recent Reward:** {training_summary['mean_recent_reward']:.3f}")
                st.write(f"**Mean Recent Length:** {training_summary['mean_recent_length']:.1f}")
                st.write(f"**Q-table Shape:** {training_summary['q_table_shape']}")
            
            # Download options
            st.subheader("Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Download Evaluation Report"):
                    report_json = json.dumps(interpretability_report, indent=2, default=str)
                    st.download_button(
                        label="Download JSON",
                        data=report_json,
                        file_name="evaluation_report.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("🤖 Download Model"):
                    agent.save_model("temp_model.npz")
                    with open("temp_model.npz", "rb") as f:
                        st.download_button(
                            label="Download NPZ",
                            data=f.read(),
                            file_name="trained_agent.npz",
                            mime="application/octet-stream"
                        )
            
            with col3:
                if st.button("📈 Download Training Data"):
                    training_data = pd.DataFrame(agent.training_history)
                    csv = training_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="training_history.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
