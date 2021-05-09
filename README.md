<h1 align="center">keytotext</h1>

[![pypi Version](https://img.shields.io/pypi/v/keytotext.svg?logo=pypi&logoColor=white)](https://pypi.org/project/keytotext/)
[![Downloads](https://static.pepy.tech/personalized-badge/keytotext?period=total&units=none&left_color=grey&right_color=orange&left_text=Pip%20Downloads)](https://pepy.tech/project/keytotext)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gagan3012/keytotext/blob/master/Examples/K2T.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/gagan3012/keytotext/UI/app.py)
[![API Call](https://img.shields.io/badge/-FastAPI-red?logo=fastapi&labelColor=white)]()
[![Docker Call](https://img.shields.io/badge/-Docker%20Image-blue?logo=docker&labelColor=white)](https://hub.docker.com/r/gagan30/keytotext)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models%20on%20Hub-yellow)](https://huggingface.co/models?filter=keytotext)
[![Documentation Status](https://readthedocs.org/projects/keytotext/badge/?version=latest)](https://keytotext.readthedocs.io/en/latest/?badge=latest)


![keytotext (1)](https://user-images.githubusercontent.com/49101362/116334480-f5e57a00-a7dd-11eb-987c-186477f94b6e.png)

Idea is to build a model which will take keywords as inputs and generate sentences as outputs. 

## Model:

Keytotext is based on the Amazing T5 Model: [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models%20on%20Hub-yellow)](https://huggingface.co/models?filter=keytotext)

- `k2t`: [Model](https://huggingface.co/gagan3012/k2t)
- `k2t-tiny`: [Model](https://huggingface.co/gagan3012/k2t-tiny)
- `k2t-base`: [Model](https://huggingface.co/gagan3012/k2t-base)

Training Notebooks can be found in the [`Training Notebooks`](https://github.com/gagan3012/keytotext/tree/master/notebooks) Folder

## Usage:

Example usage: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gagan3012/keytotext/blob/master/Examples/K2T.ipynb)

Example Notebooks can be found in the [`Notebooks`](https://github.com/gagan3012/keytotext/tree/master/examples) Folder

```shell script
pip install keytotext
```

![carbon (3)](https://user-images.githubusercontent.com/49101362/116220679-90e64180-a755-11eb-9246-82d93d924a6c.png)


## UI:

UI: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/gagan3012/keytotext/UI/app.py)

```shell script
pip install streamlit-tags
```
This uses a custom streamlit component built by me: [GitHub](https://github.com/gagan3012/streamlit-tags)

![image](https://user-images.githubusercontent.com/49101362/116162205-fc042980-a6fd-11eb-892e-8f6902f193f4.png)

## API:

API: [![API Call](https://img.shields.io/badge/-Open%20with%20FastAPI-red?logo=fastapi&labelColor=white)](http://localhost:8000/api?data=[%22India%22,%22Capital%22,%22New%20Delhi%22])
[![Docker Call](https://img.shields.io/badge/-Docker%20Image-blue?logo=docker&labelColor=white)](https://hub.docker.com/r/gagan30/keytotext)

The API is hosted in the Docker container and it can be run quickly.
Follow instructions below to get started

```shell script
docker pull gagan30/keytotext

docker run -dp 8000:8000 gagan30/keytotext
```

This will start the api at port 8000 visit the url below to get the results as below:
```
http://localhost:8000/api?data=["India","Capital","New Delhi"]
```

![k2t_json](https://user-images.githubusercontent.com/49101362/117046515-c56e7600-acde-11eb-8a20-7e1ab5f0de02.png)

Note: The Hosted API is only available on demand
## BibTex:

To quote keytotext please use this citation

```bibtex
@misc{bhatia, 
      title={keytotext},
      url={https://github.com/gagan3012/keytotext}, 
      journal={GitHub}, 
      author={Bhatia, Gagan}
}
```
