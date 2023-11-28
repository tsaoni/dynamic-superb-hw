import os
from argparse import Namespace

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["use_wandb"] = "true"

experiment = "sample"
options = [
    ("EnvironmentalSoundClassification", "EnvironmentalSoundClassification_ESC50-Animals"), 
    ("SpeechTextMatching", "SpeechTextMatching_LJSpeech"), 
    ("NoiseSNRLevelPrediction", "NoiseSNRLevelPrediction_VCTK_MUSAN-Gaussian"), 
    ("SpoofDetection", "SpoofDetection_ASVspoof2017"), 
    ("DialogueActClassification", "DialogueActClassification_DailyTalk"), 
    ("SpeakerCounting", "SpeakerCounting_LibriTTS-TestClean"),
    #("SourceDetection", "SourceDetection_mb23-music_caps_4sec_wave_type"), 
    #("SourceDetection", "SourceDetection_mb23-music_caps_4sec_wave_type_continuous"), 
]

assert experiment in [
    "sample", "preprocess", "espnet", "multimodel-llama", "test", "salmonn", "nextgpt", "ltu"
]

# test
if experiment.startswith("test"):
    COMMAND = "python test.py"

elif experiment.startswith("sample"):
    task_root = "dynamic_superb/benchmark_tasks"
    ensemble_type = "v1"
    if ensemble_type != "no":
        save_root = f"results/{ensemble_type}"
    else: save_root = "results/SALMONN"

    loop = 1
    if loop:
        for i in range(0, 6):
            opt = i
            mode = "gen-predict"
            task, instance = options[opt]
            json_path = os.path.join(task_root, task, instance, "instance.json")
            save_path = os.path.join(save_root, task, instance, "prediction.json")
            download_dir = os.environ.get("HF_DATASETS_CACHE")
            assert mode in ["gen-predict", "analysis"]
            assert ensemble_type in ["no", "v1"]
            COMMAND = f"python sample.py \
                        --mode {mode} \
                        --task {task} \
                        --instance {instance} \
                        --ensemble_type {ensemble_type} \
                        --json_path {json_path} \
                        --save_path {save_path} \
                        --download_dir {download_dir} \
                        "
            os.system(COMMAND)
        exit(-1)
    opt = -1
    mode = "create-instance"
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
    opt = -1
    os.chdir("api/preprocess")
    for i in range(1):
        opt = i
        task, instance = options[opt]
        json_path = os.path.join(task_root, task, instance, "instance.json")
        save_path = os.path.join(save_root, task, instance)
        multi_uttrs = 1
        COMMAND = f"python process_instance.py \
                    --json_path {json_path} \
                    --save_dir {save_path} "
        os.system(COMMAND)
    exit(-1)
    if multi_uttrs: COMMAND += "--multi_uttrs "

# inference
elif experiment.startswith("espnet"):
    mode = "infer"
    opt = 0
    assert mode in ["data-prepare", "infer", "store-result", "all"]
    task, instance = options[opt]
    os.environ["DATA_DIR"] = f"/tmp2/yuling/dynamic-superb-hw/data/{task}"
    if mode in ["data-prepare"]:
        os.chdir("espnet-whisper/egs2/dynamic_superb")
        COMMAND = """
            bash run.sh --stage 1 --stop_stage 5 ;
            python create_bigsuperb_prompt.py ;
            bash run.sh --stage 10 --stop_stage 10 ;
        """
        os.system(COMMAND)
    if mode in ["infer"]:
        os.chdir("espnet-whisper/egs2/dynamic_superb")
        train_dir = "asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual"
        COMMAND = f"""
            cp valid.acc.ave.pth exp/{train_dir}/ ;
            bash run.sh --stage 11 --stop_stage 11 --ngpu 1 ;
            bash run.sh --stage 12 --stop_stage 13 --ngpu 1 ;
        """ #12,13
        # bash download_checkpoint.sh ;
        # bash run.sh --stage 11 --stop_stage 11 --ngpu 0 ;
        # mv valid.acc.ave.pth exp/{train_dir}/ ;
        os.system(COMMAND)
    if mode in ["store-result"]:
        save_root = "results/espnet"
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
    data_root = "../data"
    os.chdir("multimodal-llama")
    mode = "val"
    assert mode in ["train", "val"]
    if mode == "val":
        eval_ckpt = 4
        n_epoch = 10
        ckpt_name = f"model-{n_epoch}.pth"
        out_ckpt_pair = [
            #("aud", None), 
            ("cnt", f"output/SpeechTextMatching_LibrispeechTrainClean100/ckpts/{ckpt_name}"), 
            ("deg", f"output/NoiseSNRLevelPredictionGaussian_VoxcelebMusan/ckpts/{ckpt_name}"), 
            ("para", f"output/SpoofDetection_Asvspoof2017/ckpts/{ckpt_name}"), 
            ("sem", f"output/DialogueActClassification_DailyTalk/ckpts/{ckpt_name}"), 
            ("spk", f"output/SpeakerCounting_LibrittsTrainClean100/ckpts/{ckpt_name}"), 
        ]
        for i in range(6):
            opt = i
            task, instance = options[opt]
            data_path = os.path.join(data_root, task)
            if eval_ckpt is not None:
                output_dir, ckpt_path = f"whisper-{out_ckpt_pair[eval_ckpt][0]}-{n_epoch}", out_ckpt_pair[eval_ckpt][1]
                dataset_list = f"{out_ckpt_pair[eval_ckpt][0]}_test_dataset.txt"
            else:
                output_dir, ckpt_path = "whisper", "ckpts/dynamic-superb/whisper-llama-latest.pth"
                dataset_list = f"test_dataset.txt"
            COMMAND = f"echo '{instance}' > data/{dataset_list} ; \
                        python dynamicsuperb_inference.py \
                        --model_path {ckpt_path} \
                        --encoder_type whisper \
                        --output_dir ../results/{output_dir}/{task} \
                        --dataset_list data/{dataset_list} \
                        --data_path {data_path} \
                        --llama_path ckpts/llama_model_weights "
            os.system(COMMAND)
        exit(-1)
    elif mode == "train":
        task = "BigSuperbPrivate"
        instances = os.listdir(f"../data/{task}")
        instance = instances[4] #'\n'.join(instances)
        epochs = 1000
        exp = instance #"total"
        output_dir = f"./output/{exp}/ckpts"
        log_dir = f"./output/{exp}/log"
        dataset_list = f"data/{exp}_train_dataset.txt"
        COMMAND = f"echo '{instance}' > {dataset_list} ; \
                python dynamicsuperb_finetune.py \
                    --output_dir {output_dir} \
                    --log_dir {log_dir} \
                    --epochs {epochs} \
                    --dataset_list {dataset_list} "

elif experiment.startswith("salmonn"):
    paths = dict(
        ckpt_path="salmonn_v1.pth", 
        whisper_path="models--openai--whisper-large-v2/snapshots/696465c62215e36a9ab3f9b7672fe7749f1a1df5",
        beats_path="BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        vicuna_path="models--lmsys--vicuna-13b-v1.1/snapshots/1acf26f93742fafe91562253ec0e5d94e40a8bea",
    )
    for pth_name, pth in paths.items():
        paths[pth_name] = os.path.join("ckpts", pth)
    opt = 0
    os.chdir("SALMONN")
    for i in range(1, 6):
        opt = i
        task, instance = options[opt]
        wav_path = f"../data/{task}/{instance}"
        output_path = f"../results/SALMONN/{task}"
        COMMAND = f"python3 cli_inference.py \
                    --ckpt_path {paths['ckpt_path']} \
                    --whisper_path {paths['whisper_path']} \
                    --beats_path {paths['beats_path']} \
                    --vicuna_path {paths['vicuna_path']} \
                    --wav_path {wav_path} \
                    --task {task} \
                    --instance {instance} \
                    --output_path {output_path} "
        os.system(COMMAND)
    exit(-1)

elif experiment.startswith("nextgpt"):
    os.chdir("NExT-GPT/code")
    #mode = "train"
    #assert mode in ["train", "val"]
    COMMAND = f"python inference.py"

elif experiment.startswith("ltu"):
    os.chdir("ltu")
    for i in range(2, 6):
        opt = i
        task, instance = options[opt]
        wav_path = f"../data/{task}/{instance}"
        output_path = f"../results/ltu/{task}"
        COMMAND = f"python ltu2_api_demo.py \
                    --wav_path {wav_path} \
                    --task {task} \
                    --instance {instance} \
                    --output_path {output_path} "
        os.system(COMMAND)
    exit(-1)

os.system(COMMAND)