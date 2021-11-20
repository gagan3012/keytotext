# for TPU trainign with tf.data.Dataset

import torch
import torch_xla.core.xla_model as xm
from keytotext import trainer, make_dataset

train_df = make_dataset('common_gen', split='train')
eval_df = make_dataset('common_gen', split='validation')
test_df = make_dataset('common_gen', split='test')

model = trainer()
model.from_pretrained(model_name="t5-base")
model.train(train_df=train_df, test_df=eval_df, batch_size=4, max_epochs=10, use_gpu=False,tpu_cores=8)
print(model.evaluate(test_df=test_df))
model.upload(hf_username="gagan3012", model_name="k2t-base")
