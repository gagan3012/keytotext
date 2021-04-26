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
    st.sidebar.markdown(
        '''
        ## Keytotext UI 
        *For additional questions and inquiries, please contact **Gagan Bhatia** via [LinkedIn](
        https://www.linkedin.com/in/gbhatia30/) or [Github](https://github.com/gagan3012).*
        ''')

    keywords = st_tags(
        label='## Enter Keywords:',
        text='Press enter to add more',
        value=['India', 'wedding', 'Food'],
        maxtags=4,
        key='1')

    st.sidebar.selectbox("Select the model you would like to run", ['k2t', 'k2t-base', 'k2t-tiny'])
    if st.button("Get Answer"):
        text = generate(keywords)
        st.write("# Generated Sentence:")
        st.write("## {}".format(text))


if __name__ == '__main__':
    display()
