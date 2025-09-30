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


def run_optimized(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str, more_kwargs: str | dict = {}):
    start_time = time.time()
    
    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    
    # Aggressive optimizations for speed
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    setup_kwargs.update({
        "sampling_timesteps": 15,  # Aggressive reduction from 50
        "max_size": 384,  # Lower resolution
        "smo_k_s": 5,  # Reduce smoothing
        "smo_k_d": 1,  # Minimal smoothing
    })
    
    run_kwargs = more_kwargs.get("run_kwargs", {})
    
    print(f"Setup time: {time.time() - start_time:.2f}s")
    
    SDK.setup(source_path, output_path, **setup_kwargs)
    
    # Optimize audio processing
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)
    
    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)
    
    processing_start = time.time()
    print(f"Audio processing time: {time.time() - processing_start:.2f}s")
    
    # Use offline mode for better batching
    aud_feat = SDK.wav2feat.wav2feat(audio)
    SDK.audio2motion_queue.put(aud_feat)
    SDK.close()
    
    # Optimize ffmpeg command
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v libx264 -preset ultrafast -crf 23 -c:a aac "{output_path}"'
    print(cmd)
    os.system(cmd)
    
    total_time = time.time() - start_time
    print(f"Total generation time: {total_time:.2f}s")
    print(f"Speed ratio: {len(audio)/16000:.1f}s audio -> {total_time:.1f}s generation")
    print(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_Ampere_Plus", help="path to trt data_root")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl", help="path to cfg_pkl")
    parser.add_argument("--audio_path", type=str, help="path to input wav")
    parser.add_argument("--source_path", type=str, help="path to input image")
    parser.add_argument("--output_path", type=str, help="path to output mp4")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode optimizations")
    args = parser.parse_args()

    # Enable GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # init sdk
    data_root = args.data_root
    cfg_pkl = args.cfg_pkl
    SDK = StreamSDK(cfg_pkl, data_root)

    # input args
    audio_path = args.audio_path
    source_path = args.source_path
    output_path = args.output_path

    # Fast mode optimizations
    if args.fast:
        more_kwargs = {
            "setup_kwargs": {
                "sampling_timesteps": 20,
                "max_size": 512,
                "smo_k_s": 7,  # Reduce smoothing
                "smo_k_d": 1,  # Reduce smoothing
            }
        }
        run_optimized(SDK, audio_path, source_path, output_path, more_kwargs)
    else:
        run_optimized(SDK, audio_path, source_path, output_path)