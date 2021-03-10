import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelWithLMHead
import re


st.set_page_config(
    page_title="Text Generation Using Keywords",
    layout="wide",
    initial_sidebar_state="expanded", )

@st.cache(suppress_st_warning=True,ttl=1000)
def generate(keywords,temp,top_p):
    tokenizer = AutoTokenizer.from_pretrained("gagan3012/keytotext-small")
    model = AutoModelWithLMHead.from_pretrained("gagan3012/keytotext-small")
    text = str(keywords)
    text = text.replace(',', ' | ')
    text = text.replace("'", "")
    text = text.replace('[', '')
    text = text.replace(']', '')
    texts = text.split(".")
    result = ""
    for txt in texts:
        input_ids = tokenizer.encode("WebNLG:{} </s>".format(txt),
                                     return_tensors="pt")
        outputs = model.generate(input_ids, max_length=1024, temperature=temp, top_p=top_p)
        result += tokenizer.decode(outputs[0])
    result = re.sub('<pad>|</s>', "", result)
    return result


