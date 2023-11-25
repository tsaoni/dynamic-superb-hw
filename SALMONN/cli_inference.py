# Copyright (2023) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json
import torch
import argparse
from collections import defaultdict
from model import SALMONN

def accuracy(origin: str, predict: str):
    return 1 if origin.lower() == predict.lower() else 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--whisper_path", type=str, default=None)
    parser.add_argument("--beats_path", type=str, default=None)
    parser.add_argument("--vicuna_path", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--instance", type=str, default=None)
    parser.add_argument("--wav_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--low_resource", action='store_true', default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    model = SALMONN(
        ckpt=args.ckpt_path,
        whisper_path=args.whisper_path,
        beats_path=args.beats_path,
        vicuna_path=args.vicuna_path,
        low_resource=args.low_resource
    )
    model.to(args.device)
    model.eval()
    metadata = json.load(open(os.path.join(args.wav_path, "metadata.json")))
    predictions = []
    pred_stat, label_stat = defaultdict(lambda: 0), defaultdict(lambda: 0)
    counter = 0
    acc, n_total = 0, len(metadata)
    for filename, data in metadata.items():
        if counter == 10: break
        counter += 1
        print("=====================================")
        wav_path = os.path.join(args.wav_path, filename)
        prompt_temp = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
        prompt = prompt_temp.format(instruction=data["instruction"])
        try:
            print("Output:")
            # for environment with cuda>=117
            with torch.cuda.amp.autocast(dtype=torch.float16):
                response = model.generate(wav_path, prompt=prompt)[0]
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
        except Exception as e:
            print(e)
            if args.debug:
                import pdb; pdb.set_trace()
    import pdb 
    pdb.set_trace()
    results = {
        "accuracy": acc / n_total, 
        "pred_count": pred_stat,
        "label_count": label_stat,
        "predictions": predictions, 
    }
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, f"{args.instance}.json"), "w") as f:
        json.dump(results, f, indent=2)
