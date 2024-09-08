import streamlit as st

from pages import dashboard

def main():
    st.title('Dynamic Shortest-path Interdiction')
    dashboard.display()

if __name__ == "__main__":
    main()
