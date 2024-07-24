__all__ = ['RealTimeLoss']
import os, sys
from pathlib import Path
sys.path.append(str((Path(os.path.abspath(__file__)).parents[3])))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from dashboard.monitoring.email_setting import EmailRealLoss

ROOT_DIR = str((Path(os.path.abspath(__file__)).parents[3]))
LOG_PATH = os.path.join(ROOT_DIR, "logs", "training_loss", "training_loss.csv")
SAVED_MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "torch", "mobilenetv3_tmp.pth")

MAX_POINTS = 20
ALERT_THRESHOLD = 0.45
MIN_EMAIL = 3
MAX_EPOCH = 5

class RealTimeLoss(EmailRealLoss):
    def __init__(self, log_path=None, max_points=None, max_epoch=None):
        self.log_path = log_path
        if self.log_path is None:
            self.log_path = LOG_PATH
        
        self.max_points = MAX_POINTS
        self.email_sent = False
        self.max_epoch = MAX_EPOCH

    def display_comparison_results(self, placeholders=None):
        
        if os.path.exists(self.log_path):
            df = pd.read_csv(self.log_path)
            if not df.empty:
                recent_df = df.tail(self.max_points)
                fig = go.Figure()

                fig.add_trace(go.Scatter(x=recent_df['Epoch'], y=recent_df['Loss'], mode='lines+markers', name='LOSS'))
                fig.update_layout(title='Model Training Loss Over Recent Epoch', xaxis_title='Step', yaxis_title='Cumulative Success Rate')
                # 
                if placeholders:
                    placeholders[0].plotly_chart(fig, use_container_width=True)
                    epoch = recent_df['Epoch'].iloc[-1]
                    loss = recent_df['Loss'].iloc[-1]
                    placeholders[1].write(f'saved_model_path: {SAVED_MODEL_PATH}')
                    if epoch == self.max_epoch and not self.email_sent:
                        self.send_email_alert('Model B', loss=loss, epoch=epoch, placeholder=placeholders[2])
                    
                else:
                    # st.plotly_chart(fig)
                    st.plotly_chart(fig)
                    st.write(recent_df)
            else:
                placeholders[0].write("Not yet one epoch. Please waiting")
            
        else:
            placeholders[0].write("No log data found. Please run the monitoring first.")

    def app(self):
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Select a page", ["Real-time Training Loss"])
        
        if page == "Real-time Training Loss":
            st.sidebar.title("Data Source Selection")
            if st.sidebar.button("Update"):
                if st.sidebar.button("Acknowledge Alert"):
                    self.email_sent = False
                    st.session_state.script_running = False

                placeholder1 = st.empty()
                placeholder2 = st.empty()
                placeholder3 = st.empty()

                while True:
                    self.display_comparison_results(placeholders=[placeholder1, placeholder2, placeholder3])
                    time.sleep(5)

    

# if __name__ == '__main__':
#     seg = RealTimeSegmentation()
#     seg.update_data()
#     # main()
    # st.write("Real-time segmentation accuracy comparison for TFLite model.")
