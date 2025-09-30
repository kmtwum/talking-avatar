import librosa
import math
import os
import numpy as np
import random
import torch
import pickle
import time

from stream_pipeline_offline import StreamSDK


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def run_fast(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str, more_kwargs: str | dict = {}):
    start_time = time.time()
    
    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    
    # Conservative optimizations to avoid segfault
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    setup_kwargs.update({
        "sampling_timesteps": 35,  # Moderate reduction from 50
        "smo_k_d": 2,  # Slight reduction from 3
    })
    
    run_kwargs = more_kwargs.get("run_kwargs", {})
    
    SDK.setup(source_path, output_path, **setup_kwargs)
    
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)
    
    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)
    
    # Use offline mode (more stable)
    aud_feat = SDK.wav2feat.wav2feat(audio)
    SDK.audio2motion_queue.put(aud_feat)
    SDK.close()
    
    # Fast ffmpeg encoding
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v libx264 -preset fast -crf 23 -c:a aac "{output_path}"'
    print(cmd)
    os.system(cmd)
    
    total_time = time.time() - start_time
    print(f"Generation time: {total_time:.2f}s for {len(audio)/16000:.1f}s audio")
    print(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_Ampere_Plus")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl")
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # Enable safe GPU optimizations
    torch.backends.cudnn.benchmark = True
    
    SDK = StreamSDK(args.cfg_pkl, args.data_root)
    
    # Safe optimization parameters
    more_kwargs = {
        "setup_kwargs": {
            "sampling_timesteps": 35,
            "smo_k_d": 2,
        }
    }
    
    run_fast(SDK, args.audio_path, args.source_path, args.output_path, more_kwargs)