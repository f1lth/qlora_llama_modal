import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

def train():
    train_dataset = load_dataset("pubmed_qa", 'pqa_labeled', split="train")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-13b-hf", 
        load_in_4bit=True, torch_dtype=torch.float16, device_map="auto"
    )

    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_int8_training(model)
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4, 
        optimizer='adamw_torch',
        logging_steps=100,
        learning_rate = 2e-4,
        fp16=True,
        warmup_ratio=0.1,
        lr_scheudler_type='linear',
        num_train_epochs=1,
        save_strategy='epoch',
        push_to_hub=True
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field='question',
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.push_to_hub()

train()
