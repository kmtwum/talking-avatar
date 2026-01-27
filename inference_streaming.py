import librosa
import math
import os
import numpy as np
import torch
import pickle
import time

from stream_pipeline_online import StreamSDK


def run_streaming(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str):
    """Optimized streaming inference for video calls"""
    
    # Streaming optimizations - force online mode
    setup_kwargs = {
        "sampling_timesteps": 15,
        "max_size": 384,
        "online_mode": True,  # Force online mode
        "smo_k_s": 3,
        "smo_k_d": 1,
    }
    
    SDK.setup(source_path, output_path, **setup_kwargs)
    
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)
    SDK.setup_Nd(N_d=num_f)
    
    # Stream audio in chunks for lower latency
    chunk_size = (2, 3, 1)  # Smaller chunks = lower latency
    audio = np.concatenate([np.zeros((chunk_size[0] * 640,), dtype=np.float32), audio], 0)
    split_len = int(sum(chunk_size) * 0.04 * 16000) + 80
    
    for i in range(0, len(audio), chunk_size[1] * 640):
        audio_chunk = audio[i:i + split_len]
        if len(audio_chunk) < split_len:
            audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
        SDK.run_chunk(audio_chunk, chunk_size)
    
    SDK.close()
    
    # Fast encoding for streaming
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v libx264 -preset ultrafast -tune zerolatency -crf 28 -c:a aac "{output_path}"'
    os.system(cmd)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_Ampere_Plus")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl")
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    
    SDK = StreamSDK(args.cfg_pkl, args.data_root)
    run_streaming(SDK, args.audio_path, args.source_path, args.output_path)