import streamlit as st
from multiapp import MultiApp
from home import home
# from monitoring.plot_success_rate.realtime_success_rate import RealTimeGraph
from monitoring.plot_success_rate import RealTimeSuccessRate
from monitoring.training_graph import RealTimeLoss
from monitoring.result_comparison import ImageComparator
from monitoring.shadow_test import RealTimeAccuracyComparison

def main():

    # Initialize session state to track the active page
    if 'active_page' not in st.session_state:
        st.session_state.active_page = "Home"

    # Function to update active page
    app = MultiApp()

    # Add all your application here
    app.add_app("Home", home.app)
    app.add_app("Real-Time Graph", RealTimeSuccessRate().app)
    app.add_app("Real-Time Loss", RealTimeLoss().app)
    app.add_app("Real-Time Shadow Test", RealTimeAccuracyComparison().app)
    app.add_app("Result Comparison", ImageComparator().app)
    # The main app
    app.run()


if __name__ == "__main__":
    main()