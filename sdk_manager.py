"""
SDK Manager - Singleton pattern for keeping StreamSDK loaded in memory.

This eliminates the subprocess overhead of spawning a new Python process
for each video generation request.
"""

import threading
from typing import Optional
import os


class SDKManager:
    """Thread-safe singleton manager for StreamSDK instances."""
    
    _instance: Optional["SDKManager"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    # SDK instances
    _offline_sdk = None
    _streaming_sdk = None
    
    # Configuration paths
    _cfg_pkl: str = "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
    _data_root: str = "/app/checkpoints/ditto_trt_Ampere_Plus"
    _online_cfg_pkl: str = "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
    
    def __new__(cls) -> "SDKManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def configure(cls, cfg_pkl: str = None, data_root: str = None, online_cfg_pkl: str = None):
        """Configure SDK paths before initialization."""
        if cfg_pkl:
            cls._cfg_pkl = cfg_pkl
        if data_root:
            cls._data_root = data_root
        if online_cfg_pkl:
            cls._online_cfg_pkl = online_cfg_pkl
    
    @classmethod
    def get_offline_sdk(cls):
        """
        Get or create the offline StreamSDK instance.
        
        The offline SDK is used for standard video generation where
        the complete audio is available upfront.
        """
        instance = cls()
        
        with cls._lock:
            if cls._offline_sdk is None:
                print("[SDKManager] Loading offline SDK...")
                from stream_pipeline_offline import StreamSDK
                cls._offline_sdk = StreamSDK(cls._cfg_pkl, cls._data_root)
                print("[SDKManager] Offline SDK loaded successfully")
            
            return cls._offline_sdk
    
    @classmethod
    def get_streaming_sdk(cls):
        """
        Get or create the streaming StreamSDK instance.
        
        The streaming SDK is used for chunked fMP4 output where
        video segments are yielded as they are generated.
        """
        instance = cls()
        
        with cls._lock:
            if cls._streaming_sdk is None:
                print("[SDKManager] Loading streaming SDK...")
                # Import will be updated once we create stream_pipeline_streaming.py
                from stream_pipeline_streaming import StreamingSDK
                cls._streaming_sdk = StreamingSDK(cls._online_cfg_pkl, cls._data_root)
                print("[SDKManager] Streaming SDK loaded successfully")
            
            return cls._streaming_sdk
    
    @classmethod
    def warmup(cls):
        """
        Pre-load SDK on application startup.
        
        Call this in FastAPI's startup event to avoid cold start latency
        on the first request.
        """
        print("[SDKManager] Warming up SDKs...")
        
        try:
            # Pre-load the offline SDK (most commonly used)
            cls.get_offline_sdk()
            print("[SDKManager] Warmup complete")
        except Exception as e:
            print(f"[SDKManager] Warmup failed: {e}")
            raise
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if any SDK has been initialized."""
        return cls._offline_sdk is not None or cls._streaming_sdk is not None
    
    @classmethod
    def cleanup(cls):
        """
        Cleanup SDK resources.
        
        Called on application shutdown to free GPU memory.
        """
        with cls._lock:
            cls._offline_sdk = None
            cls._streaming_sdk = None
            cls._initialized = False
            print("[SDKManager] Cleaned up")


# Convenience function for direct import
def get_sdk():
    """Get the default offline SDK instance."""
    return SDKManager.get_offline_sdk()
