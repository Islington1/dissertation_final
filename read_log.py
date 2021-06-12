import streamlit as st


def print_log(fileName):
    with open(fileName, 'r') as f:
        #print(f.read())
        st.text(f.read())