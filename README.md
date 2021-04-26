# keytotext

Idea is to build a model which will take keywords as inputs and generate sentences as outputs. 

### Model:

Two Models have been built: 

- Using T5-base size = 850 MB can be found here: https://huggingface.co/gagan3012/keytotext
- Using T5-small size = 230 MB can be found here: https://huggingface.co/gagan3012/keytotext-small
- Updated model: https://huggingface.co/gagan3012/k2t

#### Usage:

```python
from transformers import AutoTokenizer, AutoModelWithLMHead
  
tokenizer = AutoTokenizer.from_pretrained("gagan3012/k2t")

model = AutoModelWithLMHead.from_pretrained("gagan3012/k2t")
```

This uses a custom streamlit component built by me: [GitHub](https://github.com/gagan3012/streamlit-tags)

```
pip install streamlit-tags
```

The installation can also be found on [**PyPi**](https://pypi.org/project/streamlit-tags/)

### Example: 

['India', 'Wedding']  -> We are celebrating today in New Delhi with three wedding anniversary parties.
