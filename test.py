#import jiwer
import os, json
import random
import time
import torch

mode = "test"

assert mode in ["test", "spoken", "hugchat", "instance"]

if mode.startswith("test"):
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM

    cache_dir = "./NExT-GPT/ckpt/delta_ckpt/nextgpt/7b_tiva_v0"
    model_name = "ChocoWu/nextgpt_7b_tiva_v0"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=cache_dir, 
                                                    load_in_8bit=True, torch_dtype=torch.float16)
    # Load model directly
    # from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    # processor = AutoProcessor.from_pretrained("openai/whisper-large-v2", cache_dir=cache_dir)
    # model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v2", cache_dir=cache_dir)
    exit(-1)
    from jiwer import wer

    while True:
        reference = input("ref: ")
        hypothesis = input("hypo: ")

        error = wer(reference, hypothesis)
        import pdb 
        pdb.set_trace()

elif mode.startswith("instance"):
    hf_root = "DynamicSuperb"
    origin_task_root = "../dynamic-superb/dynamic_superb/benchmark_tasks"
    task_root = "dynamic_superb/benchmark_tasks"
    task = "SourceDetection"
    instance = "SourceDetection_mb23-music_caps_4sec_wave_type_continuous"
    target_path = os.path.join(task_root, task, instance)
    os.makedirs(target_path, exist_ok=True)
    json_path = os.path.join(target_path, "instance.json")
    metadata = {
        "name": instance, 
        "description": "", 
        "keywords": "", 
        "metrics": [
            "accuracy", 
        ], 
        "path": os.path.join(hf_root, instance), 
        "version": "d9498d5174914ad15327519d3ee7c60e0d64c2b8", 
    }
    json.dump(metadata, open(json_path, "w"), indent=4)
    readme_paths = [os.path.join(task_root, task), os.path.join(task_root, task, instance)]
    for p in readme_paths:
        open(os.path.join(p, "README.md"), "w").write("")

elif mode.startswith("spoken"):
    data_root = "Spoken-SQuAD"
    dataset_root = f"{data_root}/Spoken-SQuAD_audio"
    train_files = os.listdir(f"{dataset_root}/train_wav")
    dev_files = os.listdir(f"{dataset_root}/dev_wav")
    dev_json_file = f"{data_root}/spoken_test-v1.1.json"
    dev_data = json.load(open(dev_json_file))
    import pdb 
    pdb.set_trace()

elif mode.startswith("hugchat"):
    from hugchat import hugchat
    from hugchat.login import Login

    n_data = 10
    email = os.environ.get("HUGCHAT_EMAIL")
    passwd = os.environ.get("HUGCHAT_PWD")
    sign = Login(email, passwd)
    cookies = sign.login()

    # Save cookies to the local directory
    cookie_path_dir = "./cookies_snapshot"
    sign.saveCookiesToDir(cookie_path_dir)

    # Load cookies when you restart your program:
    # sign = login(email, None)
    # cookies = sign.loadCookiesFromDir(cookie_path_dir) # This will detect if the JSON file exists, return cookies if it does and raise an Exception if it's not.

    # Create a ChatBot
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())  # or cookie_path="usercookies/<email>.json"
    instrs = [
        "According to the given two audio signals, decide whether they are from the same source. Please answer with `match` or `mismatch`.", 
        "Listen to the following audio signals, determine if they are from the same music. Your answer should be either `match` or `mismatch`.", 
        "The musical sound signals are extracted from some tracks, response `match` if they are from the same track, otherwise please answer `mismatch`.", 
    ]
    queries = [
        "please give me responses such as: {instr}", 
        "please rephrase a sentence that is similar to: {instr}", 
        "generate the sentence like: {instr}", 
    ]

    for i in range(n_data):
        instr, query = random.sample(instrs, 1)[0], random.sample(queries, 1)[0]
        query_result = chatbot.query(query.format(instr=instr))
        print(query_result.text) # or query_result.text or query_result["text"]
        time.sleep(20)
#python -m fastchat.model.apply_delta --base  delta_ckpt/models--lmsys--vicuna-13b-delta-v0/snapshots/3082b9f4b63712f003c3d751a4a737606c2e68f7 --target pretrained_ckpt/vicuna_ckpt/13b_v0/  --delta pretrained_ckpt/vicuna_ckpt/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c2f3ec81aac798ae26dcc57799a994dfbf521496


# python /home/yuling/miniconda3/lib/python3.10/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py  --input_dir  llama-2-13b  --model_size 13B --output_dir tmp 
