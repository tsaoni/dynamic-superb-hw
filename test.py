import jiwer
import os, json
import random
import time

mode = "hugchat"

assert mode in ["test", "spoken", "hugchat"]

if mode.startswith("test"):
    from jiwer import wer

    while True:
        reference = input("ref: ")
        hypothesis = input("hypo: ")

        error = wer(reference, hypothesis)
        import pdb 
        pdb.set_trace()
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