#!/usr/bin/env python3
"""
TensorRT optimization using existing conversion script
"""

import os
import sys
import subprocess
import argparse


def optimize_trt_engines(onnx_dir: str, output_dir: str, gpu_arch: str = "A10"):
    """Use existing cvt_onnx_to_trt.py script"""
    
    script_path = "./scripts/cvt_onnx_to_trt.py"
    
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found")
        print("Please run from the ditto-talkinghead root directory")
        return False
        
    cmd = [sys.executable, script_path, "--onnx_dir", onnx_dir, "--trt_dir", output_dir]
    
    print(f"Converting ONNX to TensorRT engines...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ TensorRT engines created in {output_dir}")
        print(f"\nUse with: --data_root {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to convert: {e}")
        print("\nFallback: Use existing engines from ./checkpoints/ditto_trt_Ampere_Plus")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", default="./checkpoints/ditto_onnx")
    parser.add_argument("--output_dir", default="./checkpoints/ditto_trt_A10_optimized")
    parser.add_argument("--gpu", default="A10")
    
    args = parser.parse_args()
    optimize_trt_engines(args.onnx_dir, args.output_dir, args.gpu)