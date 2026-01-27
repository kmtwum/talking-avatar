"""
Test client for WebSocket streaming endpoint.

Usage:
    # Standard chunk mode
    python test_socket_client.py [--url ws://localhost:8000/ws/generate]
    
    # Word-by-word streaming (tests aggregation)
    python test_socket_client.py --word-stream "Hello world, this is a test. How are you today?"
    
    # Disable aggregation
    python test_socket_client.py --no-aggregate --text "Short" "chunks" "here"
"""

import asyncio
import argparse
import json
import time
from typing import List


async def test_socket_streaming(
    url: str, 
    chunks: List[str], 
    avatar: str = "sunny", 
    size: int = 256,
    aggregate: bool = True,
    aggregate_min_chars: int = 50,
    aggregate_max_chars: int = 500,
    aggregate_timeout: float = 1.5,
    delay_between_chunks: float = 0.1
):
    """
    Test the WebSocket streaming endpoint.
    
    Args:
        url: WebSocket URL
        chunks: List of text chunks to send
        avatar: Avatar name
        size: Output video size
        aggregate: Enable chunk aggregation
        aggregate_min_chars: Minimum chars before flush
        aggregate_max_chars: Max chars before force flush
        aggregate_timeout: Timeout in seconds
        delay_between_chunks: Delay between sending chunks
    """
    try:
        import websockets
    except ImportError:
        print("Please install websockets: pip install websockets")
        return
    
    print(f"Connecting to {url}...")
    print(f"Aggregation: {'ENABLED' if aggregate else 'DISABLED'}")
    if aggregate:
        print(f"  min_chars={aggregate_min_chars}, max_chars={aggregate_max_chars}, timeout={aggregate_timeout}s")
    print(f"Sending {len(chunks)} chunks with {delay_between_chunks}s delay")
    print()
    
    async with websockets.connect(url) as ws:
        start_time = time.time()
        
        # Send SESSION_START with aggregation config
        print(f"[{time.time() - start_time:.2f}s] Sending SESSION_START...")
        await ws.send(json.dumps({
            "type": "SESSION_START",
            "avatar": avatar,
            "size": size,
            "tts_preference": "coqui",
            "aggregate_chunks": aggregate,
            "aggregate_min_chars": aggregate_min_chars,
            "aggregate_max_chars": aggregate_max_chars,
            "aggregate_timeout": aggregate_timeout,
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
        
        # Send text chunks with delays
        for i, text in enumerate(chunks):
            display_text = text[:30] + "..." if len(text) > 30 else text
            print(f"[{time.time() - start_time:.2f}s] Chunk {i}: '{display_text}'")
            await ws.send(json.dumps({
                "type": "SPEECH_CHUNK",
                "seq": i,
                "text": text
            }))
            
            await asyncio.sleep(delay_between_chunks)
            
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
            message = await asyncio.wait_for(ws.recv(), timeout=60.0)
            
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
                    print(f"[{time.time() - start_time:.2f}s] Status: chunks_processed={msg.get('chunks_processed')}")
                    
                elif msg_type == "SESSION_COMPLETE":
                    print(f"[{time.time() - start_time:.2f}s] Session complete!")
                    print(f"  Duration: {msg.get('total_duration_ms')}ms")
                    print(f"  Chunks processed: {msg.get('chunks_processed')}")
                    print(f"  Frames generated: {msg.get('frames_generated')}")
                    break
                    
                elif msg_type == "ERROR":
                    print(f"[{time.time() - start_time:.2f}s] ERROR: {msg.get('message')} ({msg.get('code')})")
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


def split_into_words(text: str) -> List[str]:
    """Split text into individual words, preserving punctuation."""
    words = []
    current_word = ""
    
    for char in text:
        if char in ' \t\n':
            if current_word:
                words.append(current_word)
                current_word = ""
        else:
            current_word += char
            
    if current_word:
        words.append(current_word)
        
    return words


def main():
    parser = argparse.ArgumentParser(
        description="Test WebSocket streaming endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard test with default sentences
  python test_socket_client.py
  
  # Word-by-word streaming (demonstrates aggregation)
  python test_socket_client.py --word-stream "Hello! This is a test."
  
  # Compare with aggregation disabled
  python test_socket_client.py --word-stream "Hello there!" --no-aggregate
  
  # Custom aggregation settings
  python test_socket_client.py --min-chars 30 --timeout 1.0
        """
    )
    parser.add_argument("--url", default="ws://localhost:8000/ws/generate", help="WebSocket URL")
    parser.add_argument("--avatar", default="sunny", help="Avatar name")
    parser.add_argument("--size", type=int, default=256, help="Output video size")
    parser.add_argument("--text", nargs="+", help="Text chunks to send")
    parser.add_argument("--word-stream", type=str, help="Stream text word-by-word (tests aggregation)")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between chunks in seconds")
    
    # Aggregation settings
    parser.add_argument("--no-aggregate", action="store_true", help="Disable chunk aggregation")
    parser.add_argument("--min-chars", type=int, default=50, help="Minimum chars before flush")
    parser.add_argument("--max-chars", type=int, default=500, help="Maximum chars before force flush")
    parser.add_argument("--timeout", type=float, default=1.5, help="Aggregation timeout in seconds")
    
    args = parser.parse_args()
    
    # Determine chunks
    if args.word_stream:
        # Word-by-word streaming mode
        chunks = split_into_words(args.word_stream)
        print(f"Word streaming mode: {len(chunks)} words")
    elif args.text:
        chunks = args.text
    else:
        # Default test chunks
        chunks = [
            "Hello! This is the first chunk of my speech.",
            "Now I'm continuing with the second part.",
            "And here's the final sentence of my message.",
        ]
    
    asyncio.run(test_socket_streaming(
        url=args.url,
        chunks=chunks,
        avatar=args.avatar,
        size=args.size,
        aggregate=not args.no_aggregate,
        aggregate_min_chars=args.min_chars,
        aggregate_max_chars=args.max_chars,
        aggregate_timeout=args.timeout,
        delay_between_chunks=args.delay,
    ))


if __name__ == "__main__":
    main()

