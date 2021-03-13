import streamlit as st
from streamlit_tags import st_tags
from transformers import AutoTokenizer, AutoModelWithLMHead
import re
import torch

st.set_page_config(
    page_title="Text Generation Using Keywords",
    layout="wide",
    initial_sidebar_state="expanded", )


@st.cache(suppress_st_warning=True, ttl=1000)
def generate(keywords, temp, top_p):
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


def display():
    st.write('# Using AI to Generate Sentences from Keywords')
    st.sidebar.markdown(
        '''
        ## This is a demo of a text to text generation model trained with T5 to generate Sentences from Keywords
        *For additional questions and inquiries, please contact **Gagan Bhatia** via [LinkedIn](
        https://www.linkedin.com/in/gbhatia30/) or [Github](https://github.com/gagan3012).*
        ''')
    st.sidebar.write('## Options:')

    keys = st.sidebar.slider(label='Number of keywords', min_value=1, max_value=15, value=3, step=1)
    top_p = st.sidebar.slider(label='Top k', min_value=0.0, max_value=40.0, value=1.0, step=1.0)
    temp = st.sidebar.slider(label='Temperature', min_value=0.1, max_value=1.0, value=1.0, step=0.05)
    st.sidebar.markdown(
        '''
        `Number of Keywords:` number of keywords given\n
        `Temperature:` Float value controlling randomness in boltzmann distribution. Lower temperature results in less random completions. As the temperature approaches zero, the model will become deterministic and repetitive. Higher temperature results in more random completions.\n
        `Top k:` Integer value controlling diversity. 1 means only 1 word is considered for each step (token), resulting in deterministic completions, while 40 means 40 words are considered at each step. 0 (default) is a special setting meaning no restrictions. 40 generally is a good value.
        ''')

    keywords = st_tags('Enter Keyword:', 'Press enter to add more', ['One', 'Two', 'Three'])

    if st.button("Get Answer"):
        text = generate(keywords, temp, top_p)
        st.write("# Generated Sentence:")
        st.write("## {}".format(text))


if __name__ == '__main__':
    display()