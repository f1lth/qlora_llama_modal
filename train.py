from modal import Image, Stub, gpu, method, web_endpoint

stub = Stub("finetuneMyLLM")
    #Load the model
import torch
import transformers

model_id = 'mosaicml/mpt-7b'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'triton'

model = transformers.AutoModelForCausalLM.from_pretrained(
  model_id,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True
)

image = 
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate",
        "diffusers[torch]>=0.15.1",
        "ftfy",
        "torchvision",
        "transformers~=4.25.1",
        "triton",
        "safetensors",
    )
    .pip_install(
        "torch==2.0.1+cu117",
        find_links="https://download.pytorch.org/whl/torch_stable.html",
    )
    .pip_install("xformers", pre=True)
    .run_function(
        download_models,
        secrets=[Secret.from_name("huggingface-secret")],
    )
)
stub.image = image








from datasets import load_dataset

dataset_artificial = load_dataset("pubmed_qa", "pqa_artificial")

dataset_labeled = load_dataset("pubmed_qa", "pqa_labeled")

#format the data
import json

def format_dataset_forqlora(dataset):
    qlora_dataset = {
        'data': []
    }

    for example in dataset['train']:
        paragraphs = []
        context = example['context']['contexts'][0]
        paragraph = {
            'context': context,
            'qas': []
        }
        paragraphs.append(paragraph)

        question = example['question']
        answer = example['long_answer']
        qas = {
            'question': question,
            'id': str(example['pubid']),
            'answers': [
                {
                    'text': answer,
                    'answer_start': context.find(answer)
                }
            ]
        }
        paragraphs[0]['qas'].append(qas)

        data = {
            'title': f"Document {example['pubid']}",
            'paragraphs': paragraphs
        }
        qlora_dataset['data'].append(data)

    return qlora_dataset

# Example usage
qlora_formatted_dataset = format_dataset_forqlora(dataset_labeled)

# Save the formatted dataset to a file
with open('qlora_dataset.json', 'w') as f:
    json.dump(qlora_formatted_dataset, f)

