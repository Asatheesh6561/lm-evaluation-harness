from datasets import Dataset, DatasetDict
import json
from huggingface_hub import HfApi, HfFolder

with open("wmdp-translate.json", "r") as f:
    data = json.load(f)

data_dict = {key: [d[key] for d in data] for key in data[0].keys()}
dataset = Dataset.from_dict(data_dict)
dataset_dict = DatasetDict({"translate_test": dataset})

api = HfApi()
token = HfFolder.get_token()
api.create_repo("asatheesh/wmdp-translate", token=token, exist_ok=True)

dataset_dict.push_to_hub("asatheesh/wmdp-translate")
