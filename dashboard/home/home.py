import os
import streamlit as st
from home.home_text import summary_tmp

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def read_markdown_file(markdown_file):
    with open(markdown_file, 'r', encoding='utf-8') as file:
        return file.read()
    
def app():
    st.title("Team8. R.I.P")
    st.write("""
    Use the sidebar to navigate to different pages.
    """)
    st.markdown(summary_tmp)

    st.write("""
    README
    """)
    readme = read_markdown_file(os.path.join(ROOT_DIR, "README.md"))
    st.markdown(readme)
    
