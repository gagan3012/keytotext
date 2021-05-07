import streamlit as st
from streamlit_tags import st_tags
from keytotext import pipeline
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

st.set_page_config(
    page_title="Text Generation Using Keywords",
)


@st.cache(suppress_st_warning=True,
          ttl=1000,
          show_spinner=False)
def generate_model(model="k2t"):
    nlp = pipeline(model)
    return nlp


def display():
    st.write('# Keytotext UI')
    st.markdown(
        '''
        *For additional questions and inquiries, please contact **Gagan Bhatia** via [LinkedIn](
        https://www.linkedin.com/in/gbhatia30/) or [Github](https://github.com/gagan3012).*
        ''')

    mt = st.slider(label="Number of Keywords you would like to enter:",
                   min_value=1,
                   max_value=10,
                   value=3,
                   step=1)

    model_name = st.selectbox(label='Select model:',
                         options=['k2t','k2t-base'])

    keywords = st_tags(label='## Enter Keywords:',
                       text='Press enter to add more',
                       value=['India', 'Capital', 'New Delhi'],
                       maxtags=mt,
                       key='keywords')

    if st.button("Generate text"):
        with st.spinner("Connecting the Dots..."):
            model = generate_model(model=model_name)
            text = model(keywords)
        st.write("# Generated Sentence:")
        st.write("## {}".format(text))


if __name__ == '__main__':
    display()
