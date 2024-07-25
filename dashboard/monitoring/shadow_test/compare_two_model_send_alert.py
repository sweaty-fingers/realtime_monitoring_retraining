__all__ = ['RealTimeAccuracyComparison']
import os, sys
from pathlib import Path
sys.path.append(str((Path(os.path.abspath(__file__)).parents[3])))

import threading
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from dashboard.monitoring.email_setting import EmailRealAB

ROOT_DIR = str((Path(os.path.abspath(__file__)).parents[3]))
LOG_PATH = os.path.join(ROOT_DIR, "logs", "comparing_ab_model", "compaing_result.csv")
COMPARING_SCRIPT_PATH = os.path.join(ROOT_DIR, "running_code", "compare_performance.py")

MAX_POINTS = 20
COMPARING_STEPS = 3

pd.options.mode.chained_assignment = None
# 이메일 설정 (예시: Gmail을 사용하는 경우)
class RealTimeAccuracyComparison(EmailRealAB):
    def __init__(self):
        self.email_sent = False
        self.max_points = MAX_POINTS
        self.comparing_step = 3
        self.log_path = LOG_PATH

    def display_comparison_results(self, placeholders=None):
        if os.path.exists(self.log_path):
            df = pd.read_csv(self.log_path)

            if not df.empty:
                recent_df = df.tail(self.max_points)
                recent_df['compare'] = recent_df['new_accuracy'] > recent_df['old_accuracy']
                fig = go.Figure()

                # Clear existing traces
                fig.data = []

                # Add new traces
                fig.add_trace(go.Scatter(x=recent_df['step'], y=recent_df['old_accuracy'], mode='lines+markers', name='Old Model', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=recent_df['step'], y=recent_df['new_accuracy'], mode='lines+markers', name='New Model', line=dict(color='red')))

                # Fill area between Old Model and New Model
                for i in range(len(recent_df['step']) - 1):
                    if recent_df['old_accuracy'].iloc[i] > recent_df['new_accuracy'].iloc[i]:
                        fig.add_trace(go.Scatter(
                            x=[recent_df['step'].iloc[i], recent_df['step'].iloc[i+1], recent_df['step'].iloc[i+1], recent_df['step'].iloc[i]],
                            y=[recent_df['new_accuracy'].iloc[i], recent_df['new_accuracy'].iloc[i+1], recent_df['old_accuracy'].iloc[i+1], recent_df['old_accuracy'].iloc[i]],
                            fill='toself',
                            fillcolor='rgba(0, 0, 255, 0.2)',
                            line=dict(color='rgba(0,0,0,0)'),
                            showlegend=False  # 레이블 표시 방지
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=[recent_df['step'].iloc[i], recent_df['step'].iloc[i+1], recent_df['step'].iloc[i+1], recent_df['step'].iloc[i]],
                            y=[recent_df['old_accuracy'].iloc[i], recent_df['old_accuracy'].iloc[i+1], recent_df['new_accuracy'].iloc[i+1], recent_df['new_accuracy'].iloc[i]],
                            fill='toself',
                            fillcolor='rgba(255, 0, 0, 0.2)',
                            line=dict(color='rgba(0,0,0,0)'),
                            showlegend=False  # 레이블 표시 방지
                        ))

                    # if recent_df['continual_success_count'].iloc[i] >= self.comparing_step and not self.email_sent:
                    if recent_df['compare'].tail(self.comparing_step).sum() >= self.comparing_step and not self.email_sent:
                        self.send_email_alert('Model NEW', placeholder=placeholders[1])

                fig.update_layout(title='Real-time Accuracy Comparison', xaxis_title='Time', yaxis_title='Accuracy')
                # self.chart_container.plotly_chart(fig)
                placeholders[0].plotly_chart(fig)
            
            else:
                placeholders[0].write("Not yet one epoch. Please waiting")
            
        else:
            placeholders[0].write("No log data found. Please run the monitoring first.")
    
    def app(self):
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Select a page", ["Real-time Model Comparing"])
        
        if page == "Real-time Model Comparing":
            st.sidebar.title("Data Source Selection")
            if st.sidebar.button("Update"):
                if st.sidebar.button("Acknowledge Alert"):
                    self.email_sent = False
                    st.session_state.script_running = False

                placeholder1 = st.empty()
                placeholder2 = st.empty()

                while True:
                    self.display_comparison_results(placeholders=[placeholder1, placeholder2])
                    time.sleep(5)

