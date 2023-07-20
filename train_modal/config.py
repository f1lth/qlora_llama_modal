from modal import Image, SharedVolume, Stub
import random
from typing import Optional
from pathlib import Path
'''
# fine-tune the model on a dataset: pubmed_qa 
# via lora training on a gpu 
'''
# move this to env vars eventually
VOL_MOUNT_PATH = Path("/vol")
MULTI_WORKSPACE_SLACK_APP = False
WANDB_PROJECT = ""
MODEL_PATH = "/model"

def download_models():
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

    model_name = "openlm-research/open_llama_7b_400bt_preview"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(MODEL_PATH)

with open('./requirements.txt') as f:
    requirements = f.read().splitlines()

def download_dataset():
    from datasets import load_dataset
    dataset = load_dataset("pubmed_qa", 'pqa_labeled', split="train")
    dataset.save_to_disk("/dataset")

# setup virtual environment to send to modal
image = (
    Image.micromamba()
    .micromamba_install(
        "cudatoolkit=11.7",
        "cudnn=8.1.0",
        "cuda-nvcc",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    .pip_install(requirements
    )
    .run_function(download_models)
    .run_function(download_dataset)
    .pip_install("wandb==0.15.0")
)

# create vm
stub = Stub(name="pubmed_lora", image=image)

# ? whats this for

stub.slack_image = (
    Image.debian_slim()
    .pip_install("slack-sdk", "slack-bolt")
    .apt_install("wget")
    .run_commands(
        "sh -c 'echo \"deb http://apt.postgresql.org/pub/repos/apt bullseye-pgdg main\" > /etc/apt/sources.list.d/pgdg.list'",
        "wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -",
    )
    .apt_install("libpq-dev")
    .pip_install("psycopg2")
)

output_vol = SharedVolume(cloud="gcp").persist("slack-finetune-vol")

def generate_prompt(user, input, output=""):
    # change this to match your dataset
    return f"""You are {user}, employee at a fast-growing startup. Below is an input conversation that takes place in the company's internal Slack. Write a response that appropriately continues the conversation.

        ### Input:
        {input}

        ### Response:
        {output}"""


def user_data_path(user: str, team_id: Optional[str] = None) -> Path:
    return VOL_MOUNT_PATH / (team_id or "data") / user / "data.json"


def user_model_path(user: str, team_id: Optional[str] = None, checkpoint: Optional[str] = None) -> Path:
    path = VOL_MOUNT_PATH / (team_id or "data") / user
    if checkpoint:
        path = path / checkpoint
    return path
