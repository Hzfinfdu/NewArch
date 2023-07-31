from torchscale.architecture.retnet import RetNetDecoder
import torch
from config import retnet_config, training_arguments
from transformers import Trainer, GPT2TokenizerFast
import datasets
import utils
from utils import print_rank_0

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')  # We use gpt2 tokenizer
tokenizer.pad_token = tokenizer.eos_token
dataset = datasets.load_dataset('./openwebtext')

dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=retnet_config.max_target_positions), batched=True, num_proc=32)
dataset = dataset.remove_columns(["text"])

if training_arguments.group_by_length:
    dataset = dataset.map(lambda x: {'length': sum(x['attention_mask'])}, batched=False, num_proc=32)

train_dataset, val_dataset = dataset['train'].train_test_split(test_size=0.0005).values()

print_rank_0(train_dataset)

model = RetNetDecoder(retnet_config).cuda()
print_rank_0("Number of Retnet parameters: ", utils.count_parameters(model))

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()