import streamlit as st
from streamlit_tags import st_tags
from keytotext import pipeline
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

st.set_page_config(
    page_title="Text Generation Using Keywords",
)
