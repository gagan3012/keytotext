# Pretrain and Fine-tune a T5 model with Flax on GCP (Approach abandoned)

This tutorial details how pretrain and fine-tune a [FlaxT5](https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_flax_t5.py) model from HuggingFace using a TPU VM available on Google Cloud.

While the code is only slightly adapted from the original HuggingFace examples for [pretraining](https://github.com/huggingface/transformers/tree/master/examples/flax/language-modeling#t5-like-span-masked-language-modeling) and [seq2seq fine-tuning](https://github.com/huggingface/transformers/tree/master/examples/flax/summarization), this repository is aimed to provide a comprehensive overview for the whole process, with a special focus on **pitfalls** due to an incorrect environment setup.

_**Why JAX/Flax?**_ Thanks to the amazing work of the HuggingFace team, a good portion of the models available in the `transformers` library are available in [Flax](https://flax.readthedocs.io/en/latest/index.html), a neural network library built on top of [JAX](https://jax.readthedocs.io/en/latest/index.html). Using Flax for pretraining on TPUs is especially convenient for two reasons:

- TPUs are optimized for training deep models, and Flax is optimized for TPUs. [A benchmark](https://github.com/huggingface/transformers/tree/master/examples/flax/language-modeling#runtime-evaluation) conducted by Huggingface showed that a BERT pretraining with Flax takes only 65% of the time needed with Pytorch/XLA to achieve comparable performances on a TPUv3-8 VM, and only 35% of the time if compared to Pytorch training on 8 V100 GPUs. Since language models are usually pretrained for days on massive amounts of text, the speedup is especially desirable here.

- Out-of-the-box compatibility with Pytorch and Tensorflow in the `transformers` framework. A Flax model can be easily converted in Pytorch, for example, by using `T5ForConditionalGeneration.from_pretrained("path/to/flax/ckpt", from_flax=True)`.

The code and instructions contained in this repository were used to pretrain the models [`gsarti/t5-base-it`](https://huggingface.co/gsarti/t5-base-it) and [`gsarti/t5-large-it`](https://huggingface.co/gsarti/t5-large-it) available on the Huggingface Hub, using ~270Gb of cleaned web-scraped Italian texts. The dataset is also made available on the Huggingface Hub under the name [`gsarti/clean_mc4_it`](https://huggingface.co/datasets/gsarti/clean_mc4_it).

## Setup on your machine

Follow the instructions detailed in the [Cloud SDK Install Guide](https://cloud.google.com/sdk/docs/install) to install the `gcloud` client on your system.

TPU VMs only have ~100Gb of disk space, which is highly unlikely to be enough to store your raw + preprocessed dataset and all model checkpoints. GCP gives a choice among different kind of disk types and limits the total disk space a user can create. At the time of writing this, for a non-free trial user in Europe without special access the limit is 250Gb for SSD (`pd-balanced`) and 2Tb overall, including SSD and HDD (`pd-standard`).

```shell
### Define your variables
export GCP_PROJECT="<YOUR_PROJECT_NAME>"
export GCP_ZONE="<YOUR_REGION>"
export GCP_TPU_NAME="<YOUR_TPU_NAME>"

# >>>>> Run this part to add a disk to your TPU VM
export GCP_DISK_NAME="<YOUR_DISK_NAME>"
export GCP_DISK_SIZE_GB=1200
export GCP_DISK_TYPE=pd-standard


gcloud beta compute disks create $GCP_DISK_NAME \
    --project=$GCP_PROJECT \
    --type=$GCP_DISK_TYPE \
    --size="${GCP_DISK_SIZE_GB}GB" \
    --zone=$GCP_ZONE
# <<<<<

# Create the TPU VM
gcloud alpha compute tpus tpu-vm create $GCP_TPU_NAME \
    --zone $GCP_ZONE \
    --project $GCP_PROJECT \
    --accelerator-type v3-8 \
    --version v2-alpha \
    # Uncomment this if a disk is used
    #--data-disk source="projects/${GCP_PROJECT}/zones/${GCP_ZONE}/disks/${GCP_DISK_NAME}" 
```

## Inside the TPUv3-8 machine

Log in inside the machine with the following command:

`gcloud alpha compute tpus tpu-vm ssh $GCP_TPU_NAME --zone $GCP_ZONE --project $GCP_PROJECT`

The system is a classic Ubuntu 20.04 with some utility libraries already available (e.g. `tmux`, `vim`). The following commands show how to setup the disk inside the machine, if you created a disk. The disk in the example is available on `/dev/sdb`, and we mount it on the `data` folder in the home:

```shell
# Check that your disk is visible and get its name
lsblk
# Mount disk in the data folder
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
sudo mkdir -p data
sudo mount -o discard,defaults /dev/sdb data
sudo chmod a+w data
# Optionally, add the automatic mounting: https://cloud.google.com/compute/docs/disks/add-persistent-disk#configuring_automatic_mounting_on_vm_restart

# Permanently export the disk as the new path for HF dataset caching
echo "export HF_DATASETS_CACHE=data" >> .bashrc
source .bashrc
```

We then setup the environment with required libraries and dependencies:

```shell
### Setup the TPUv3-8 environment
sudo apt update
sudo apt-get install python3.8-venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements.txt
sudo apt-get install git-lfs
git lfs install
```

The next step is to create a repository on the Huggingface Hub (here e.g. t5-base-it) that will contain all our files. We then train a tokenizer, select a config, and push the two JSON files in the repository we created.

Depending on the number of sentences that need to be processed, this may take quite some time (~2 hours for 10M sentences).

```shell
# Huggingface Hub variables
export HF_NAME="gagan3012"
export HF_PWD="<YOUR_HF_HUB_PASSWORD>"
export HF_TOKEN="<YOUR_HF_HUB_API_TOKEN>"
export HF_PROJECT="t5-k2t"
```

```shell
# Variables for training the tokenizer and creating the config
export VOCAB_SIZE="32000"
export N_INPUT_SENTENCES="1000000" # Num of sentences to train the tokenizer
export DATASET="common_gen" # Name of the dataset in the Huggingface Hub
export DATASET_CONFIG="full" # Config of the dataset in the Huggingface Hub 
export DATASET_SPLIT="train" # Split to use for training tokenizer and model
export TEXT_FIELD="text" # Field containing the text to be used for training
export CONFIG_TYPE="google/t5-v1_1-base" # Config that our model will use
export MODEL_PATH="data/${HF_PROJECT}" # Path to the model, e.g. here inside the mount

### Setup the project on the HF Hub
#huggingface-cli login also works
echo $HF_TOKEN > ~/.huggingface/token
huggingface-cli repo create $HF_PROJECT
git clone "https://huggingface.co/${HF_NAME}/${HF_PROJECT}"
mv $HF_PROJECT $MODEL_PATH

# Create the tokenizer and the config
python create_tokenizer.py \
    --model_dir $MODEL_PATH \
    --dataset $DATASET \
    --dataset_config $DATASET_CONFIG \
    --dataset_split $DATASET_SPLIT \
    --text_field $TEXT_FIELD \
    --vocab_size $VOCAB_SIZE \
    --input_sentence_size $N_INPUT_SENTENCES \
    --config_type $CONFIG_TYPE

# Push the tokenizer and the config to the HF Hub
cd $MODEL_PATH
git lfs track "*tfevents*"
git add .
git commit -m "Added tokenizer and config"à
# If this doesn't work, do a normal push and insert your credentials
git push "https://huggingface.co/${HF_NAME}/${HF_NAME}:${HF_PWD}@${HF_PROJECT}"
cd ..
```

The last step is to run the pretraining. The following command will train the model on the same dataset used to train the tokenizer, logging train/eval metrics to Tensorboard and pushing model checkpoints and artifacts to the Hub every `save_steps` steps:

**Important**: Since checkpoints are stored with Git LFS on the Hub, depending on the number of `save_steps` you may encur in problems with space (e.g. training a model with 1Gb ckpts for 1M steps with `save_steps=10_000` will need 100Gb of free space to avoid crashes during training). Two ways to avoid problems are:

- Use `save_steps` > (`tot_steps` / `your_available_memory`) * `ckpt_size_gb`. E.g. if we are doing 1M steps and the ckpt_size is 1GB and we only have 76Gb free, we will need save_steps higher than 1M/76Gb ~= 13158. Failing to do so will result in an interruption of the training once the memory has been exceeded.

- Use `git lfs prune --verify-remote` to periodically remove old cached checkpoints from `.git/lfs/objects` and free up some space. If done often enough this will prevent any problem of space, but requires manual intervention of the user.

```shell
# Run the training
# Adjust the parameters to your needs
# To avoid using an extra disk for storing the model, 
python run_t5_mlm_flax.py \
    --output_dir=$MODEL_PATH \
    --model_type="t5" \
    --config_name=$MODEL_PATH \
    --tokenizer_name=$MODEL_PATH \
    --preprocessing_num_workers="96" \
    --do_train --do_eval \
    --adafactor \
    --dataset_name="gsarti/clean_mc4_it" \
    --dataset_config_name="full" \
    --max_seq_length="512" \
    --per_device_train_batch_size="8" \
    --per_device_eval_batch_size="8" \
    --learning_rate="0.005" \
    --overwrite_output_dir \
    --num_train_epochs="1" \
    --logging_steps="500" \
    --save_steps="80000" \
    --eval_steps="2500" \
    --weight_decay="0.01" \
    --warmup_steps="10000" \
    --validation_split_count="15000" \
    --push_to_hub \
    #--gradient_accumulation_steps="2" # Uncomment to use gradient accumulation
    #--resume_from_checkpoint=$MODEL_PATH # Uncomment to resume from ckpt
```

## Exporting Checkpoints

Your model is now trained, congratulations! :tada: Now you may want to export your final Flax model to other frameworks. This can be done easily:

```shell
python export_checkpoint.py --model_dir $MODEL_PATH
cd $MODEL_PATH
git commit -m "Added TF and PT models"
git push
```

## Fine-tune a pretrained T5 model in Flax

*TODO*

## Useful Tips

**About GCP Billing** Except for TPU costs, which can be offset by taking part in the TRC program, the second highest billing voice is the uploading of files from the VM to another destination. This happens every save_step during training, to keep the Tensorboard and the checkpoint on the HF Hub in sync. For in-region uploads, at the time of writing this the cost is up to 0.12 USD per GB. This means that for a full pretraining of 1M steps with logging every 10k steps, with checkpoints of roughly 1GB each, the cost is roughly 12$USD. To reduce this cost simply increase the amount of save_steps in the `run_t5_mlm_flax.py` script.

**About data preprocessing** Imagine you've trained a base model and now you want to train a large variant, using the same tokenzer. Even if the tokenizer is the same, if the path of the tokenizer is changed the cache will be invalidated and everything will be recomputed. If this is the case, it's not a tragedy: some extra preprocessing time will be required. But mind that the tokenization + grouping take substantial space on disk, so if the space is not enough, you will need to delete the arrows from the previous HuggingFace dataset cache. You can easily do so with something like:

`find . -type f -size +2G -size -4G -exec ls -lah {} + | grep 'Aug 25 13' | wc -l `

As an example, knowing that my tokenization completed Aug 25 at 13:xx and that files in the cache are 3Gb in size each, this should return the value you set for `n_proc`. You can get the filenames of the old preprocessing caches by removing the `wc -l` pipe and delete them all to leave space for the new ones.
