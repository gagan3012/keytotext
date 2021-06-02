import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
import time
import glob
import re
import xml.etree.ElementTree as ET
from IPython.display import HTML, display

# WebNLG: https://gitlab.com/shimorina/webnlg-dataset.git
# DART: https://github.com/Yale-LILY/dart.git

files = []
dirc = [
    "/webnlg-dataset/release_v2.1/xml/train/**/*.xml",
    "/webnlg-dataset/release_v3.0/en/train/**/*.xml",
    "/webnlg-dataset/webnlg_challenge_2017/train/**/*.xml",
    "/dart/data/v1.1.1/dart-v1.1.1-full-train.xml",
]
for dir in dirc:
    file = glob.glob("{}".format(dir), recursive=True)
    files.append(file)

triple_re = re.compile("(\d)triples")
data_dct = {}
for file in files:
    tree = ET.parse(file)
    root = tree.getroot()
    triples_num = int(triple_re.findall(file)[0])
    for sub_root in root:
        for ss_root in sub_root:
            strutured_master = []
            unstructured = []
            for entry in ss_root:
                unstructured.append(entry.text)
                strutured = [triple.text for triple in entry]
                strutured_master.extend(strutured)
            unstructured = [
                i for i in unstructured if i.replace("\n", "").strip() != ""
            ]
            strutured_master = strutured_master[-triples_num:]
            strutured_master_str = (" && ").join(strutured_master)
            data_dct[strutured_master_str] = unstructured
    print(file)
mdata_dct = {"prefix": [], "input_text": [], "target_text": []}
for st, unst in data_dct.items():
    for i in unst:
        mdata_dct["prefix"].append("webNLG")
        mdata_dct["input_text"].append(st)
        mdata_dct["target_text"].append(i)

df1 = pd.DataFrame(mdata_dct)

train_df = pd.read_csv("NLGDataset.csv", index_col=[0])
train_df = train_df.iloc[:73424, :]
train_df = train_df.sample(frac=1)
batch_size = 8
num_of_batches = int(len(train_df) / batch_size)

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")


def progress(loss, value, max=100):
    return HTML(
        """ Batch loss :{loss}
      <progress    
value='{value}'max='{max}',style='width: 100%'>{value}
      </progress>
             """.format(
            loss=loss, value=value, max=max
        )
    )


tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
# moving the model to GPU
model.to(dev)

optimizer = Adafactor(
    model.parameters(),
    lr=1e-3,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False,
)


def trainer(num_of_epochs):
    model.train()

    loss_per_10_steps = []
    for epoch in range(1, num_of_epochs + 1):
        print("Running epoch: {}".format(epoch))

        running_loss = 0

        out = display(progress(1, num_of_batches + 1), display_id=True)
        evaluation_start = time.time()

        for i in range(num_of_batches):
            inputbatch = []
            labelbatch = []
            new_df = train_df[i * batch_size : i * batch_size + batch_size]
            for indx, row in new_df.iterrows():
                input = row["input_text"] + "</s>"
                labels = row["target_text"] + "</s>"
                inputbatch.append(input)
                labelbatch.append(labels)
            inputbatch = tokenizer.batch_encode_plus(
                inputbatch, padding=True, max_length=400, return_tensors="pt"
            )["input_ids"]
            labelbatch = tokenizer.batch_encode_plus(
                labelbatch, padding=True, max_length=400, return_tensors="pt"
            )["input_ids"]
            inputbatch = inputbatch.to(dev)
            labelbatch = labelbatch.to(dev)

            # clear out the gradients of all Variables
            optimizer.zero_grad()

            # Forward propogation
            outputs = model(input_ids=inputbatch, labels=labelbatch)
            loss = outputs.loss
            loss_num = loss.item()
            logits = outputs.logits
            running_loss += loss_num
            if i % 10 == 0:
                loss_per_10_steps.append(loss_num)
            out.update(progress(loss_num, i, num_of_batches + 1))

            # calculating the gradients
            loss.backward()

            # updating the params
            optimizer.step()

        running_loss = running_loss / int(num_of_batches)
        evaluation_total = time.time() - evaluation_start
        print("Training time:", evaluation_total)
        print("Epoch: {} , Running loss: {}".format(epoch, running_loss))
