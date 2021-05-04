import streamlit as st
from streamlit_tags import st_tags
from keytotext import pipeline
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

st.set_page_config(
    page_title="Text Generation Using Keywords",
)

@st.cache(allow_output_mutation=True)
def modelextract(model="k2t"):
    pipe = pipeline(model)
    return pipe

nlp = modelextract()

@st.cache(suppress_st_warning=True,
          ttl=1000,
          show_spinner=False)
def generate(keywords, model="k2t"):
    return nlp(keywords)


def display():
    st.write('# Keytotext UI')
    st.markdown(
        '''
        *For additional questions and inquiries, please contact **Gagan Bhatia** via [LinkedIn](
        https://www.linkedin.com/in/gbhatia30/) or [Github](https://github.com/gagan3012).*
        
        This keyword entry method is called streamlit tags and it also built by me.
        ''')

    mt = st.slider(label="Number of Keywords you would like to enter:",
                   min_value=1,
                   max_value=10,
                   value=3,
                   step=1)

    model = st_tags(label='## Select model:',
                    text='',
                    value=['k2t'],
                    maxtags=1,
                    key='Model')

    keywords = st_tags(label='## Enter Keywords:',
                       text='Press enter to add more',
                       value=['India', 'Capital', 'New Delhi'],
                       maxtags=mt,
                       key='keywords')

    if st.button("Generate text"):
        with st.spinner("Connecting the Dots..."):
            text = generate(keywords=keywords,
                            model=model[0])
        st.write("# Generated Sentence:")
        st.write("## {}".format(text))


if __name__ == '__main__':
    display()
