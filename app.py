import streamlit as st
from streamlit_tags import st_tags
from keytotext import pipeline

st.set_page_config(
    page_title="Text Generation Using Keywords",
    layout="wide",
    initial_sidebar_state="expanded", )


@st.cache(suppress_st_warning=True, ttl=1000)
def generate(keywords, model="k2t"):
    nlp = pipeline(model)
    return nlp(keywords)


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

    model = st.selectbox(label="Select the model you would like to run" ,
                         options = ['k2t', 'k2t-base', 'k2t-tiny'])

    keywords = st_tags(
        label='## Enter Keywords:',
        text='Press enter to add more',
        value=['India', 'wedding', 'Food'],
        maxtags=mt,
        key='1')

    if st.button("Generate text"):
        text = generate(keywords=keywords,
                        model=model)
        st.write("# Generated Sentence:")
        st.write("## {}".format(text))


if __name__ == '__main__':
    display()
