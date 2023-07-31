import time
from torchscale.architecture.config import RetNetConfig
from transformers import GPT2TokenizerFast, TrainingArguments, DataCollatorForLanguageModeling


tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')  # We use gpt2 tokenizer
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to be the eos token

retnet_config = RetNetConfig(**{
    'decoder_embed_dim': 768,
    'decoder_retention_heads': 3,
    'decoder_ffn_embed_dim': 1536,
    'decoder_layers': 12,
    'decoder_normalize_before': True,
    'activation_fn': 'gelu',
    'dropout': 0.0,
    'drop_path_rate': 0.0,
    'activation_dropout': 0.0,
    'no_scale_embedding': True,
    'layernorm_embedding': False,
    'moe_freq': 0,
    'moe_top1_expert': False,
    'moe_expert_count': 0,
    'moe_gating_use_fp32': True,
    'moe_eval_  capacity_token_fraction': 0.25,
    'moe_second_expert_policy': 'random',
    'moe_normalize_gate_prob_before_dropping': False,
    'use_xmoe': False,
    'rel_pos_buckets': 0,
    'max_rel_pos': 0,
    'deepnorm': False,
    'subln': True,
    'multiway': False,
    'share_decoder_input_output_embed': True,
    'max_target_positions': 2048,
    'no_output_layer': False,
    'layernorm_eps': 1e-05,
    'chunkwise_recurrent': False,
    'recurrent_chunk_size': 512,
    'vocab_size': tokenizer.vocab_size,
    'checkpoint_activations': False,
    'fsdp': False,
    'ddp_rank': 0,
    'xpos_rel_pos': False,
    'xpos_scale_base': 512,
    'pad_token_id': tokenizer.eos_token_id,
})

training_arguments = TrainingArguments(
    output_dir=f'./results_{time.time()}',  # output directory
    overwrite_output_dir=True,
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=375,  # number of warmup steps for learning rate scheduler
    weight_decay=0.05,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=10,
    save_steps=20000,
    save_total_limit=8,
    evaluation_strategy='steps',
    eval_steps=20000,
    dataloader_num_workers=4,
    gradient_accumulation_steps=2,
    fp16=True,
    fp16_opt_level='O2',
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-06,
    lr_scheduler_type='cosine',
    learning_rate=0.0005,
    group_by_length=True,
    report_to='wandb',
    run_name='retnet_share_emb_unemb',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    length_column_name='length',
    gradient_checkpointing=True,
)




