"""
Test client for WebSocket streaming endpoint.

Usage:
    python test_socket_client.py [--url ws://localhost:8000/ws/generate]
"""

import asyncio
import argparse
import json
import time
from typing import List


async def test_socket_streaming(url: str, chunks: List[str], avatar: str = "sunny", size: int = 256):
    """
    Test the WebSocket streaming endpoint.
    
    Args:
        url: WebSocket URL
        chunks: List of text chunks to send
        avatar: Avatar name
        size: Output video size
    """
    try:
        import websockets
    except ImportError:
        print("Please install websockets: pip install websockets")
        return
    
    print(f"Connecting to {url}...")
    
    async with websockets.connect(url) as ws:
        start_time = time.time()
        
        # Send SESSION_START
        print(f"[{time.time() - start_time:.2f}s] Sending SESSION_START...")
        await ws.send(json.dumps({
            "type": "SESSION_START",
            "avatar": avatar,
            "size": size,
            "tts_preference": "coqui"
        }))
        
        # Wait for SESSION_STARTED
        response = await ws.recv()
        msg = json.loads(response)
        print(f"[{time.time() - start_time:.2f}s] Received: {msg}")
        
        if msg.get("type") != "SESSION_STARTED":
            print(f"Error: Expected SESSION_STARTED, got {msg}")
            return
            
        session_id = msg.get("session_id")
        print(f"[{time.time() - start_time:.2f}s] Session started: {session_id}")
        
        # Start receiving video in background
        video_task = asyncio.create_task(receive_video(ws, start_time))
        
        # Send text chunks with delays (simulating progressive speech)
        for i, text in enumerate(chunks):
            print(f"[{time.time() - start_time:.2f}s] Sending chunk {i}: '{text[:50]}...'")
            await ws.send(json.dumps({
                "type": "SPEECH_CHUNK",
                "seq": i,
                "text": text
            }))
            
            # Simulate delay between chunks
            await asyncio.sleep(0.5)
            
        # Send SESSION_END
        print(f"[{time.time() - start_time:.2f}s] Sending SESSION_END...")
        await ws.send(json.dumps({
            "type": "SESSION_END"
        }))
        
        # Wait for video to complete
        await video_task
        
        print(f"[{time.time() - start_time:.2f}s] Complete!")


async def receive_video(ws, start_time: float):
    """
    Receive video chunks and JSON messages from WebSocket.
    
    Saves video to output file.
    """
    video_data = bytearray()
    message_count = 0
    segment_count = 0
    
    try:
        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=30.0)
            
            message_count += 1
            
            if isinstance(message, bytes):
                # Binary video data
                segment_count += 1
                video_data.extend(message)
                
                if segment_count == 1:
                    print(f"[{time.time() - start_time:.2f}s] Received init segment ({len(message)} bytes)")
                elif segment_count % 10 == 0:
                    print(f"[{time.time() - start_time:.2f}s] Received {segment_count} segments ({len(video_data)} bytes total)")
                    
            else:
                # JSON message
                msg = json.loads(message)
                msg_type = msg.get("type")
                
                if msg_type == "STATUS":
                    print(f"[{time.time() - start_time:.2f}s] Status: {msg}")
                    
                elif msg_type == "SESSION_COMPLETE":
                    print(f"[{time.time() - start_time:.2f}s] Session complete: {msg}")
                    break
                    
                elif msg_type == "ERROR":
                    print(f"[{time.time() - start_time:.2f}s] Error: {msg}")
                    break
                    
                else:
                    print(f"[{time.time() - start_time:.2f}s] Received: {msg}")
                    
    except asyncio.TimeoutError:
        print(f"[{time.time() - start_time:.2f}s] Timeout waiting for data")
        
    # Save video
    if video_data:
        output_path = f"output_socket_test_{int(time.time())}.mp4"
        with open(output_path, "wb") as f:
            f.write(video_data)
        print(f"[{time.time() - start_time:.2f}s] Saved video to {output_path} ({len(video_data)} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Test WebSocket streaming endpoint")
    parser.add_argument("--url", default="ws://localhost:8000/ws/generate", help="WebSocket URL")
    parser.add_argument("--avatar", default="sunny", help="Avatar name")
    parser.add_argument("--size", type=int, default=256, help="Output video size")
    parser.add_argument("--text", nargs="+", help="Text chunks to send")
    
    args = parser.parse_args()
    
    # Default test chunks
    chunks = args.text or [
        "Hello! This is the first chunk of my speech.",
        "Now I'm continuing with the second part.",
        "And here's the final sentence of my message.",
    ]
    
    asyncio.run(test_socket_streaming(
        url=args.url,
        chunks=chunks,
        avatar=args.avatar,
        size=args.size
    ))


if __name__ == "__main__":
    main()
