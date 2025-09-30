#!/usr/bin/env python3
"""
TensorRT optimization script for A10 GPU
Rebuilds engines with A10-specific optimizations
"""

import os
import subprocess
import argparse
from pathlib import Path


def optimize_trt_engines(onnx_dir: str, output_dir: str, gpu_arch: str = "A10"):
    """Optimize TensorRT engines for specific GPU architecture"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # A10-specific optimizations
    optimization_flags = [
        "--fp16",  # Enable FP16 for A10
        "--workspace=4096",  # 4GB workspace
        "--minShapes=batch:1",
        "--optShapes=batch:1", 
        "--maxShapes=batch:1",
        "--builderOptimizationLevel=5",
        "--avgTiming=8",
        "--tacticSources=+CUDNN,+CUBLAS,+CUBLAS_LT",
    ]
    
    onnx_files = [
        "appearance_extractor.onnx",
        "decoder.onnx", 
        "hubert.onnx",
        "lmdm_v0.4_hubert.onnx",
        "motion_extractor.onnx",
        "stitch_network.onnx",
        "warp_network.onnx"
    ]
    
    for onnx_file in onnx_files:
        onnx_path = os.path.join(onnx_dir, onnx_file)
        if not os.path.exists(onnx_path):
            print(f"Skipping {onnx_file} - not found")
            continue
            
        engine_name = onnx_file.replace('.onnx', '_optimized.engine')
        engine_path = os.path.join(output_dir, engine_name)
        
        cmd = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
        ] + optimization_flags
        
        print(f"Optimizing {onnx_file}...")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ Created {engine_name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to optimize {onnx_file}: {e.stderr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", default="./checkpoints/ditto_onnx")
    parser.add_argument("--output_dir", default="./checkpoints/ditto_trt_A10_optimized")
    parser.add_argument("--gpu", default="A10", choices=["A10", "A100", "RTX4090"])
    
    args = parser.parse_args()
    optimize_trt_engines(args.onnx_dir, args.output_dir, args.gpu)