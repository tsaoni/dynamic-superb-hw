# -*- coding: utf-8 -*-
# @Time    : 9/6/23 2:05 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ltu2_api_demo.py

from gradio_client import Client
import time
import os 
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

def accuracy(origin: str, predict: str):
    return 1 if origin.lower() == predict.lower() else 0

parser = argparse.ArgumentParser()

parser.add_argument("--task", type=str, default=None)
parser.add_argument("--instance", type=str, default=None)
parser.add_argument("--wav_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)

args = parser.parse_args()
metadata = json.load(open(os.path.join(args.wav_path, "metadata.json")))
predictions = []
pred_stat, label_stat = defaultdict(lambda: 0), defaultdict(lambda: 0)
acc, n_total = 0, len(metadata)
client = Client("https://yuangongfdu-ltu-2.hf.space/")
latency = 5

for filename, data in tqdm(metadata.items()):
    filename = os.path.join(args.wav_path, filename)
    prompt_temp = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
    prompt = prompt_temp.format(instruction=data['instruction'])
    try:
        response = client.predict(
            filename, "", 
            prompt, 
            "7B (Default)", api_name="/predict")
    except: response = "connection failed"
    print(response)
    predictions.append({
        "id": filename, 
        "pred": response, 
        "label": data["label"], 
        "instruction": data["instruction"], 
        "prompt": prompt, 
    })
    pred_stat[response] += 1
    label_stat[data["label"]] += 1
    acc += accuracy(data["label"], response)
    # please do add a sleep to avoid long queue, for longer audio, sleep needs to be longer
    time.sleep(latency)

results = {
    "accuracy": acc / n_total, 
    "pred_count": pred_stat,
    "label_count": label_stat,
    "predictions": predictions, 
}
os.makedirs(args.output_path, exist_ok=True)
with open(os.path.join(args.output_path, f"{args.instance}.json"), "w") as f:
    json.dump(results, f, indent=2)