# How to add new models to Keytotext

Keytotext features a pipeline that can convert keywords to sentences.

Keytotext lib is built on top of the Transformers library by HuggingFace 

Follow these steps to add your model to keytotext:

```
language: <LANGUAGE YOUR MODEL SUPPORTS>
thumbnail: "Keywords to Sentences"
tags:
- keytotext
- k2t
- Keywords to Sentences
license: "MIT"
datasets:
- < DATASETS USED>
---
```


``` python
SUPPORTED_TASKS = {
    "<MODEL NAME>": {
        "impl": NMPipeline,
        "default": {
            "model": "<MODEL NAME>",
        },
    },
}
```
