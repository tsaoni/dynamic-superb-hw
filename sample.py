import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from collections import defaultdict
import random
from typing import List, Optional, Tuple
import soundfile as sf
import io
from scipy.io.wavfile import write

from datasets import load_dataset, Dataset, DatasetDict, Audio
from tqdm import tqdm

from utils import *

# DO NOT change the following constants.
SEED = 42
SAMPLE_NUM = 3000

def main(args) -> None:

    if args.mode == "gen-predict":
        meta_data = json.load(args.json_path.open(mode="r"))
        if args.download_dir is None:
            temp_cache_dir = tempfile.mkdtemp()
            download_dir = Path(temp_cache_dir)
        else: download_dir = args.download_dir
        dataset = load_dataset(
            meta_data["path"],
            revision=meta_data["version"],
            split="test",
            cache_dir=download_dir,
        )
        # You should shuffle the dataset with the given seed (42).
        dataset = dataset.shuffle(SEED)

        predictions = []
        correct_count = 0
        model = BaseModel()
    
        if args.ensemble_type == "no":
            results = {}
            if "whisper" in args.save_path.__str__():
                filename = f"{args.save_path.parent.__str__().split('/')[-1]}.json"
                root = '/'.join(args.save_path.parent.__str__().split('/')[:-1])
                filename = os.path.join(root, filename)
                tmp = json.load(open(filename))
                print(f"precalculate Accuracy: {tmp['accuracy']:.3f}%")
                tmp = tmp['predictions']
                prefix = f"{args.save_path.__str__().split('/')[3]}_"
                for t in tmp:
                    file, pred = t['id'], t['pred']
                    results[file.replace(prefix, "")] = pred
            else:
                res_path = os.path.join(args.save_path.parent, "exp/asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual/decode_asr_fsd_asr_model_valid.acc.ave/test/text")
                lines = [x for x in open(res_path).read().split('\n') if len(x) > 0]
                prefix = f"{args.save_path.__str__().split('/')[3]}_"
                for l in lines:
                    if len(l.split()) > 1:
                        file, pred = l.split()
                    else: file, pred = l.split()[0], ""
                    results[file.replace(prefix, "")] = pred
        elif args.ensemble_type == "v1":
            results = defaultdict(dict)
            pred_files = os.listdir("./results")
            save_paths = [os.path.join("./results", p, args.task) for p in pred_files]
            for sp, pf in zip(save_paths, pred_files):
                filename = os.path.join(sp, f"{args.instance}.json")
                tmp = json.load(open(filename))
                print(f"precalculate Accuracy: {tmp['accuracy']:.3f}%")
                tmp = tmp['predictions']
                prefix = f"{args.instance}_"
                for t in tmp:
                    file, pred = t['id'], t['pred']
                    results[pf][file.replace(prefix, "")] = pred
            import pdb 
            pdb.set_trace()
        for index, example in tqdm(enumerate(dataset)):
            if index >= SAMPLE_NUM:
                break
            file_name = example["file"]
            speech_arr = example["audio"]["array"]
            speech_sr = example["audio"]["sampling_rate"]
            instr = example["instruction"]
            text_label = example["label"]
            #num1, num2 = example["number"], example["number2"]

            # if file_name in [x[0] for x in results]: continue
            # else:
            #     print("error")
            #     exit(-1)
            
            rnd_pred = results[file_name]
            pred = {"file": file_name, "prediction": rnd_pred, "label": text_label}
            if "instr" in args.save_path.__str__(): pred.update({"instruction": instr})
            predictions.append(pred)
            if rnd_pred.lower() == text_label.lower():
                correct_count += 1

        # calculate acc
        n_total = min(len(dataset), SAMPLE_NUM)
        acc = (correct_count / n_total) * 100
        print(f"Total: {n_total}")
        print(f"Accuracy: {acc:.3f}%")

        # save prediction
        args.save_path.parent.mkdir(exist_ok=True)
        with args.save_path.open(mode="w") as f:
            json.dump(predictions, f, indent=4)
    elif args.mode == "analysis":
        predictions = json.load(args.save_path.open())
        mode = "pred-dist"
        assert mode in ["label-pred-dist", "pred-dist", "corr"]
        if mode == "label-pred-dist":
            labels = [p["label"] for p in predictions]
            preds = [p["prediction"] for p in predictions]
            celltype = list(set(labels)) #+ ["other"]
            preds = [p if p.lower() in celltype else "other" for p in preds]
            labels = pd.Series(labels).value_counts(sort=False)
            preds = pd.Series(preds).value_counts(sort=False)
            df = pd.concat([labels, preds], axis=1).fillna(0).rename(columns={0: "label", 1: "predict"})
            plt = df.plot(kind='bar', rot=0)

        elif mode == "pred-dist":
            preds = [p["prediction"].lower() for p in predictions]
            df = pd.Series(preds).value_counts(sort=False)
            plt = df.plot(kind='bar', rot=0)
            for p in plt.patches:
                plt.annotate(
                    str(p.get_height()), xy=(p.get_x() + 0.15, p.get_height() + 0.15), fontsize=10
                )
        elif mode == "corr":
            import seaborn as sns
            labels = [p["label"] for p in predictions]
            preds = [p["prediction"] for p in predictions]
            celltype = list(set(labels)) #+ ["other"]
            preds = [p if p.lower() in celltype else "other" for p in preds]
            df = pd.DataFrame({"label": labels, "predict": preds}).value_counts(sort=False)
            keys = [(c1, c2) for c1 in celltype for c2 in celltype]
            for k in keys: 
                if not k in df.keys(): df[k] = 0
            df = df.reset_index()
            df = df.pivot(index='label', columns='predict')[0]
            plt = sns.heatmap(df, annot=True, fmt='g')

        img_path = os.path.join("img", args.save_path.parent)
        os.makedirs(img_path, exist_ok=True)
        img_path = os.path.join(img_path, f"{mode}.png")
        plt.get_figure().savefig(img_path)

    elif args.mode == "create-instance":
        audio_dataset = "mb23/music_caps_4sec_wave_type"
        dataset = load_dataset(audio_dataset, split="train")
        audio_dict = defaultdict(list)
        caption_lst = []
        os.makedirs("tmp", exist_ok=True)
        for index, example in tqdm(enumerate(dataset)):
            caption = example['caption']
            number, audio = example['number'], example['audio']
            save_path = f"tmp/{str(index).zfill(5)}.wav"
            sf.write(save_path, audio['array'], audio['sampling_rate'])
            audio['path'] = save_path
            #bytes_wav = bytes()
            #byte_io = io.BytesIO(bytes_wav)
            #write(byte_io, audio['sampling_rate'], audio['array'])
            #audio = byte_io.read()
            audio_dict[caption].append((number, audio))
            caption_lst.append(caption)
        caption_lst = list(set(caption_lst))
        ins_type = "match"
        task_type = "1"
        assert ins_type in ["match", "continue", "caption"]
        if ins_type == "match":
            data_dict = defaultdict(list)
            instr_options = [
                "According to the given two audio signals, decide whether two sound signals are continuous. Please answer with `yes` or `no`.", 
            ]
            if task_type == "0":
                instr_options = instructions[ins_type]
                label_options = ["match", "mismatch"]
                task_name = "DynamicSuperb/SourceDetection_mb23-music_caps_4sec_wave_type"
            elif task_type == "1":
                label_options = ["yes", "no"]
                task_name = "DynamicSuperb/SourceDetection_mb23-music_caps_4sec_wave_type_continuous"
            n_data_per_label = 1500
            counter = 0
            print("[INFO]: generating data")
            for label in tqdm(label_options):
                for i in range(n_data_per_label):
                    data_dict["instruction"].append(random.sample(instr_options, 1)[0])
                    #data_dict["file"].append(f"{str(counter).zfill(5)}.wav")
                    if label == "match":
                        a1, a2 = random.sample(audio_dict[caption_lst[i]], 2)
                        data_dict["label"].append(label)
                    elif label == "mismatch":
                        a1 = random.sample(audio_dict[caption_lst[i]], 1)[0]
                        a2 = random.sample(audio_dict[caption_lst[i + 1]], 1)[0]
                        data_dict["label"].append(label)
                    elif label == "yes":
                        a1 = audio_dict[caption_lst[counter]][0]
                        a2 = audio_dict[caption_lst[counter]][1]
                        data_dict["label"].append(label)
                    elif label == "no":
                        if len(audio_dict[caption_lst[counter]]) < 3:
                            a1 = audio_dict[caption_lst[counter]][0]
                            a2 = audio_dict[caption_lst[counter]][1]
                            data_dict["label"].append("yes")
                        else:
                            a1 = audio_dict[caption_lst[counter]][0]
                            a2 = audio_dict[caption_lst[counter]][2]
                            data_dict["label"].append(label)

                    data_dict["audio"].append(a1[-1])
                    data_dict["audio2"].append(a2[-1])
                    data_dict["file"].append(f"{label}_{i}_0.wav") #a1[-1]["path"]
                    data_dict["file2"].append(f"{label}_{i}_1.wav") #a2[-1]["path"]
                    data_dict["number"].append(a1[0])
                    data_dict["number2"].append(a2[0])
                    counter += 1
        #audio, instruction, label
        dataset = Dataset.from_dict(data_dict)
        print("[INFO]: cast to audio")
        dataset = DatasetDict({"test": dataset}).cast_column("audio", Audio()).cast_column("audio2", Audio())

        print("[INFO]: push to hub")
        dataset.push_to_hub(task_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="gen-predict")
    parser.add_argument("--ensemble_type", type=str, default="no")
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--instance", type=str, default="")
    parser.add_argument("--json_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--download_dir", type=Path, default=None)
    args = parser.parse_args()
    main(args)