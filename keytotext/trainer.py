import pandas as pd
import torch
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
import time
import sentencepiece
import glob
import os
import re
import xml.etree.ElementTree as ET

files = []
dirc = ['/webnlg-dataset/release_v2.1/xml/train/**/*.xml',
        '/webnlg-dataset/release_v3.0/en/train/**/*.xml',
        '/webnlg-dataset/webnlg_challenge_2017/train/**/*.xml',
        '/dart/data/v1.1.1/dart-v1.1.1-full-train.xml']
for dir in dirc:
    file = glob.glob("{}".format(dir), recursive=True)
    files.append(file)

triple_re=re.compile('(\d)triples')
data_dct={}
for file in files:
    tree = ET.parse(file)
    root = tree.getroot()
    triples_num=int(triple_re.findall(file)[0])
    for sub_root in root:
        for ss_root in sub_root:
            strutured_master=[]
            unstructured=[]
            for entry in ss_root:
                unstructured.append(entry.text)
                strutured=[triple.text for triple in entry]
                strutured_master.extend(strutured)
            unstructured=[i for i in unstructured if i.replace('\n','').strip()!='' ]
            strutured_master=strutured_master[-triples_num:]
            strutured_master_str=(' && ').join(strutured_master)
            data_dct[strutured_master_str]=unstructured
    print(file)
mdata_dct={"prefix":[], "input_text":[], "target_text":[]}
for st,unst in data_dct.items():
    for i in unst:
        mdata_dct['prefix'].append('webNLG')
        mdata_dct['input_text'].append(st)
        mdata_dct['target_text'].append(i)

