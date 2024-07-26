__all__ = ['RealTimeSuccessRate']
import os, sys
from pathlib import Path
sys.path.append(str((Path(os.path.abspath(__file__)).parents[3])))

import threading
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from dashboard.monitoring.email_setting import EmailRealTime

ROOT_DIR = str((Path(os.path.abspath(__file__)).parents[3]))
LOG_PATH = os.path.join(ROOT_DIR, "logs", "realtime_monitoring", "old_model_log.csv")
RETRAINING_SCRIPT_PATH = os.path.join(ROOT_DIR, "retraining", "training", "training.py")

MAX_POINTS = 30
ALERT_THRESHOLD = 0.97
MIN_EMAIL = 20

def run_script(path):
    with st.spinner("Running the script..."):
        os.system(f"python {path}")
    st.success("Script execution completed.")

class RealTimeSuccessRate(EmailRealTime):
    def __init__(self, log_path=None, email_settings=None, max_points=None, alert_threshold=None, retraining_path=None):
        self.log_path = log_path
        if self.log_path is None:
            self.log_path = LOG_PATH

        self.max_points = MAX_POINTS
        self.minumum_length_to_email = MIN_EMAIL
        self.alert_threshold = ALERT_THRESHOLD
        self.email_sent = False
        self.retraining_path = RETRAINING_SCRIPT_PATH
        # self.script_thread = None
    
    def display_comparison_results(self, placeholders=None):
        
        if os.path.exists(self.log_path):
            df = pd.read_csv(self.log_path)
            
            # if len(df) > self.max_points:
            df['moving_acc'] = df['accuracy'].rolling(window=self.max_points).mean()
            df['moving_acc'] = df['moving_acc'].fillna(df['accuracy'].expanding().mean())
            # else:
            # df['moving_acc'] = df['accuracy'].expanding().mean()

            recent_df = df.tail(self.max_points)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recent_df['step'], y=recent_df['moving_acc'], mode='lines+markers', name='Cumulative Accuracy'))
            fig.add_shape(type="line", x0=recent_df['step'].iloc[0], y0=self.alert_threshold, x1=recent_df['step'].iloc[-1], y1=self.alert_threshold,
                        line=dict(color="Red", width=2, dash="dash"))
            
            fig.update_layout(title='Accuracy (moving average) Over Recent Steps', xaxis_title='Step', yaxis_title='Accuracy')
            # 
            if placeholders:
                placeholders[0].plotly_chart(fig, use_container_width=True)
                placeholders[1].dataframe(recent_df.drop(columns=['cumulative_success_rate']).iloc[::-1])
                
                # cum_suc = recent_df.iloc[-1]['cumulative_success_rate']
                moving_acc = recent_df.iloc[-1]['moving_acc']
                if moving_acc < self.alert_threshold and not self.email_sent and len(recent_df) > self.minumum_length_to_email:
                    self.send_email_alert('Model A', moving_acc, placeholder=placeholders[2])
                    if not st.session_state.script_running:
                        st.session_state.script_running = True
                        script_thread = threading.Thread(target=run_script, args=(RETRAINING_SCRIPT_PATH,))
                        script_thread.start()
                        placeholders[3].write("Start Retraining...")
                    else:
                        placeholders[3].warning("Script is already running.")
                                
            else:
                # st.plotly_chart(fig)
                st.plotly_chart(fig)
                st.write(recent_df)
            
        else:
            placeholders[0].write("No log data found. Please run the monitoring first.")

    def app(self):
        # Initialize session state
        if 'script_running' not in st.session_state:
            st.session_state.script_running = False

        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Select a page", ["Real-time Monitoring"])
        
        if page == "Real-time Monitoring":
            st.sidebar.title("Data Source Selection")
            if st.sidebar.button("Update"):
                placeholder1 = st.empty()
                placeholder2 = st.empty()
                placeholder3 = st.empty()
                placeholder4 = st.empty()

                if st.sidebar.button("Acknowledge Alert"):
                    self.email_sent = False

                # if st.sidebar.button("Allow additional retraining"):
                #     if self.script_thread.is_alive():
                #         placeholder4.warning("Script is still running.")
                        
                #     else:
                #         st.session_state.script_running = False
                #         # self.script_thread.join()
                #         placeholder4.write("Allow retraining")

                while True:
                    self.display_comparison_results(placeholders=[placeholder1, placeholder2, placeholder3, placeholder4])
                    time.sleep(1)

    


# if __name__ == '__main__':
#     seg = RealTimeSegmentation()
#     seg.update_data()
#     # main()
    # st.write("Real-time segmentation accuracy comparison for TFLite model.")
