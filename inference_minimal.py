import librosa
import math
import os
import numpy as np
import random
import torch
import pickle

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


def run(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str, more_kwargs: str | dict = {}):
    import time
    start_time = time.time()

    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    setup_start = time.time()
    SDK.setup(source_path, output_path, **setup_kwargs)
    print(f"Setup time: {time.time() - setup_start:.2f}s")

    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)

    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    processing_start = time.time()
    online_mode = SDK.online_mode
    if online_mode:
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
    
    generation_start = time.time()
    SDK.close()
    print(f"Generation time: {time.time() - generation_start:.2f}s")

    # Use copy codec for video (no re-encoding)
    ffmpeg_start = time.time()
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    os.system(cmd)
    print(f"FFmpeg time: {time.time() - ffmpeg_start:.2f}s")
    
    total_time = time.time() - start_time
    audio_duration = len(audio) / 16000
    print(f"Total: {total_time:.2f}s for {audio_duration:.1f}s audio (ratio: {total_time/audio_duration:.2f}x)")
    print(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_Ampere_Plus", help="path to trt data_root")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl", help="path to cfg_pkl")
    parser.add_argument("--audio_path", type=str, help="path to input wav")
    parser.add_argument("--source_path", type=str, help="path to input image")
    parser.add_argument("--output_path", type=str, help="path to output mp4")
    parser.add_argument("--steps", type=int, default=25, help="sampling timesteps")
    args = parser.parse_args()

    # Enable optimizations
    torch.backends.cudnn.benchmark = True

    # Create optimized config
    more_kwargs = {
        "setup_kwargs": {
            "sampling_timesteps": args.steps,
            "max_size": 512,  # Reduce from default 1920
        }
    }

    # init sdk
    SDK = StreamSDK(args.cfg_pkl, args.data_root)
    run(SDK, args.audio_path, args.source_path, args.output_path, more_kwargs)