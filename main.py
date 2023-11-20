import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

experiment = "multimodel-llama"
options = [
    ("EnvironmentalSoundClassification", "EnvironmentalSoundClassification_ESC50-Animals"), 
    ("SpeechTextMatching", "SpeechTextMatching_LJSpeech"), 
    ("NoiseSNRLevelPrediction", "NoiseSNRLevelPrediction_VCTK_MUSAN-Gaussian"), 
    ("SpoofDetection", "SpoofDetection_ASVspoof2017"), 
    ("DialogueActClassification", "DialogueActClassification_DailyTalk"), 
    ("SpeakerCounting", "SpeakerCounting_LibriTTS-TestClean"),
    ("SourceDetection", "SourceDetection_mb23-music_caps_4sec_wave_type"), 
]

assert experiment in ["sample", "preprocess", "espnet", "multimodel-llama", "test"]

# test
if experiment.startswith("test"):
    COMMAND = "python test.py"

elif experiment.startswith("sample"):
    task_root = "dynamic_superb/benchmark_tasks"
    save_root = "results/whisper"
    loop = 0
    if loop:
        for i in range(6):
            opt = i
            mode = "gen-predict"
            task, instance = options[opt]
            json_path = os.path.join(task_root, task, instance, "instance.json")
            save_path = os.path.join(save_root, task, instance, "prediction.json")
            download_dir = os.environ.get("HF_DATASETS_CACHE")
            assert mode in ["gen-predict", "analysis"]
            COMMAND = f"python sample.py \
                        --mode {mode} \
                        --json_path {json_path} \
                        --save_path {save_path} \
                        --download_dir {download_dir} \
                        "
            os.system(COMMAND)
        exit(-1)
    opt = -1
    mode = "gen-predict"
    task, instance = options[opt]
    json_path = os.path.join(task_root, task, instance, "instance.json")
    save_path = os.path.join(save_root, task, instance, "prediction.json")
    download_dir = os.environ.get("HF_DATASETS_CACHE")
    assert mode in ["gen-predict", "analysis", "create-instance"]
    COMMAND = f"python sample.py \
                --mode {mode} \
                --json_path {json_path} \
                --save_path {save_path} \
                --download_dir {download_dir} \
                 "

# data preprocess
elif experiment.startswith("preprocess"):
    task_root = "../../dynamic_superb/benchmark_tasks"
    save_root = "../../data"
    opt = 0
    task, instance = options[opt]
    json_path = os.path.join(task_root, task, instance, "instance.json")
    save_path = os.path.join(save_root, task, instance)
    multi_uttrs = 0
    os.chdir("api/preprocess")
    COMMAND = f"python process_instance.py \
                --json_path {json_path} \
                --save_dir {save_path} "
    if multi_uttrs: COMMAND += "--multi_uttrs "

# inference
elif experiment.startswith("espnet"):
    mode = "store-result"
    opt = 5
    task, instance = options[opt]
    os.environ["DATA_DIR"] = f"/tmp2/yuling/dynamic-superb/data/{task}"
    assert mode in ["data-prepare", "infer", "store-result"]
    
    if mode == "data-prepare":
        os.chdir("espnet-whisper/egs2/dynamic_superb")
        COMMAND = """
            bash run.sh --stage 1 --stop_stage 5 ;
            python create_bigsuperb_prompt.py ;
            bash run.sh --stage 10 --stop_stage 10 ;
        """
    elif mode == "infer":
        train_dir = "asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual"
        os.chdir("espnet-whisper/egs2/dynamic_superb")
        COMMAND = f"""
            cp valid.acc.ave.pth exp/{train_dir}/ ;
            bash run.sh --stage 11 --stop_stage 11 --ngpu 1 ;
            bash run.sh --stage 12 --stop_stage 13 --ngpu 1 ;
        """ #12,13
        # bash download_checkpoint.sh ;
        # bash run.sh --stage 11 --stop_stage 11 --ngpu 0 ;
        # mv valid.acc.ave.pth exp/{train_dir}/ ;
    elif mode == "store-result":
        save_root = "results"
        res_root = "espnet-whisper/egs2/dynamic_superb"
        res_dir = ["dump", "data", "exp"]
        source_dir = [os.path.join(res_root, d) for d in res_dir]
        target_dir = [os.path.join(save_root, task, instance, d) for d in res_dir]
        os.makedirs(save_root, exist_ok=True)
        import shutil
        for s, t in zip(source_dir, target_dir):
            shutil.move(s, t)
        exit(-1)

# whisper-LLM
elif experiment.startswith("multimodel-llama"):
    opt = -1
    data_root = "../data"
    task, instance = options[opt]
    data_path = os.path.join(data_root, task)
    os.chdir("multimodal-llama")
    COMMAND = f"echo '{instance}' > data/test_dataset.txt ; \
                python dynamicsuperb_inference.py \
                --model_path ckpts/dynamic-superb/whisper-llama-latest.pth \
                --encoder_type whisper \
                --output_dir ../results/whisper/{task} \
                --dataset_list data/test_dataset.txt \
                --data_path {data_path} \
                --llama_path ckpts/llama_model_weights "


os.system(COMMAND)