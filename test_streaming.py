#!/usr/bin/env python3
import subprocess
import sys
import os

def test_streaming_inference():
    """Test streaming inference with error checking"""
    
    audio_path = "/tmp/tts_-7536771269302821817.wav"
    source_path = "/app/avatar.jpg"  # Use default avatar
    output_path = "/tmp/test_streaming.mp4"
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return False
    
    cmd = [
        "python", "/app/inference_streaming.py",
        "--audio_path", audio_path,
        "--source_path", source_path,
        "--output_path", output_path
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"Stdout: {result.stdout}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        
        if os.path.exists(output_path):
            print(f"✓ Video generated: {output_path}")
            return True
        else:
            print(f"✗ Video not generated")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Process timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    test_streaming_inference()