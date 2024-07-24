import streamlit as st

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func,
        })

    def run(self):
        app_titles = [app['title'] for app in self.apps]
        selected_title = st.sidebar.selectbox('Navigation', app_titles)

        for app in self.apps:
            if app['title'] == selected_title:
                set_page(selected_title)
                app['function']()
                break

def set_page(page):
    st.session_state.active_page = page

class MultiAppV1:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        app = st.sidebar.selectbox(
            'Navigation',
            self.apps,
            format_func=lambda app: app['title']
        )
        app['function']()
