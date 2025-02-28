import streamlit as st
from multiapp import MultiApp
from test import realstoc

app= MultiApp()
st.markdown("Starting Page")

app.add_app("realstoc",realstoc.app)

app.run()