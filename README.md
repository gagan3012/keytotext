# keytotext
[![pypi Version](https://img.shields.io/pypi/v/keytotext.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/keytotext/)
[![Downloads](https://static.pepy.tech/personalized-badge/keytotext?period=total&units=none&left_color=grey&right_color=orange&left_text=Pip%20Downloads)](https://pepy.tech/project/keytotext)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gagan3012/keytotext/blob/master/Examples/K2T.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/gagan3012/keytotext/UI/app.py)
[![API Call](https://img.shields.io/badge/-Open%20with%20FastAPI-red?logo=fastapi&labelColor=white)]()

![keytotext (1)](https://user-images.githubusercontent.com/49101362/116334480-f5e57a00-a7dd-11eb-987c-186477f94b6e.png)

Idea is to build a model which will take keywords as inputs and generate sentences as outputs. 

**Keytotext is powered by Huggingface 🤗**

## Model:

Keytotext is based on the Amazing T5 Model: 

- `k2t`: [Model](https://huggingface.co/gagan3012/k2t)
- `k2t-tiny`: [Model](https://huggingface.co/gagan3012/k2t-tiny)
- `k2t-base`: [Model](https://huggingface.co/gagan3012/k2t-base)

Training Notebooks can be found in the [`Training Notebooks`](https://github.com/gagan3012/keytotext/tree/master/Training%20Notebooks) Folder

## Usage:

Example usage: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gagan3012/keytotext/blob/master/Examples/K2T.ipynb)

Example Notebooks can be found in the [`Notebooks`](https://github.com/gagan3012/keytotext/tree/master/Examples) Folder

```
pip install keytotext
```

![carbon (3)](https://user-images.githubusercontent.com/49101362/116220679-90e64180-a755-11eb-9246-82d93d924a6c.png)


## UI:

UI: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/gagan3012/keytotext/UI/app.py)

```
pip install streamlit-tags
```
This uses a custom streamlit component built by me: [GitHub](https://github.com/gagan3012/streamlit-tags)

![image](https://user-images.githubusercontent.com/49101362/116162205-fc042980-a6fd-11eb-892e-8f6902f193f4.png)

## API:

API: [![API Call](https://img.shields.io/badge/API-Open%20API-red)]()

To run the API please run the Docker file using `docker-compose build` and then visit 

```
http://127.0.0.1/api?data=["India","Capital","New Delhi"]
```
![k2t_json](https://user-images.githubusercontent.com/49101362/117046515-c56e7600-acde-11eb-8a20-7e1ab5f0de02.png)

