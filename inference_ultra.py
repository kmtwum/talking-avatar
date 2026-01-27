import librosa
import math
import os
import numpy as np
import torch
import pickle

from stream_pipeline_offline import StreamSDK


def run_ultra(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str):
    """Ultra-fast inference for video calls"""
    
    # Ultra optimizations
    setup_kwargs = {
        "sampling_timesteps": 10,  # Very aggressive
        "max_size": 256,  # Very low resolution
        "smo_k_s": 1,  # No smoothing
        "smo_k_d": 1,
    }
    
    SDK.setup(source_path, output_path, **setup_kwargs)
    
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)
    SDK.setup_Nd(N_d=num_f)
    
    # Process offline (faster than streaming)
    aud_feat = SDK.wav2feat.wav2feat(audio)
    SDK.audio2motion_queue.put(aud_feat)
    SDK.close()
    
    # Ultra-fast encoding
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    os.system(cmd)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_Ampere_Plus")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl")
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    
    SDK = StreamSDK(args.cfg_pkl, args.data_root)
    run_ultra(SDK, args.audio_path, args.source_path, args.output_path)