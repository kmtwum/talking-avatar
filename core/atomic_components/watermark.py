"""
Watermark Overlay - Composites a PNG logo onto video frames.

Loads a PNG with alpha channel and efficiently alpha-blends it onto
each frame during video generation. Designed for minimal per-frame cost
by pre-computing the overlay at the target frame size.
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple

# Default watermark logo path
DEFAULT_WATERMARK_PATH = "/app/img/watermark.png"


class WatermarkOverlay:
    """
    Overlays a watermark/logo onto video frames.
    
    Pre-computes the alpha-blended overlay at init time for the target
    frame dimensions, making per-frame application very fast (~0.1ms).
    
    Args:
        logo_path: Path to PNG file with alpha channel
        position: Corner placement — "bottom-right", "bottom-left",
                  "top-right", or "top-left"
        scale: Logo width as fraction of frame width (0.0–1.0)
        opacity: Overall opacity multiplier (0.0–1.0)
        margin: Pixel margin from frame edge
    """
    
    def __init__(
        self,
        logo_path: str = DEFAULT_WATERMARK_PATH,
        position: str = "bottom-right",
        scale: float = 0.15,
        opacity: float = 0.6,
        margin: int = 10,
    ):
        if not os.path.exists(logo_path):
            raise FileNotFoundError(f"Watermark logo not found: {logo_path}")
        
        # Load PNG with alpha channel (BGRA)
        raw = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Could not read watermark image: {logo_path}")
        
        # Convert to RGBA
        if raw.shape[2] == 4:
            self._logo_rgba = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
        elif raw.shape[2] == 3:
            # No alpha channel — add fully opaque alpha
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
            self._logo_rgba = np.concatenate([rgb, alpha], axis=2)
        else:
            raise ValueError(f"Unexpected channel count: {raw.shape[2]}")
        
        self._position = position
        self._scale = scale
        self._opacity = opacity
        self._margin = margin
        
        # Cache for pre-computed overlay at a specific frame size
        self._cached_frame_size: Optional[Tuple[int, int]] = None
        self._cached_logo_rgb: Optional[np.ndarray] = None
        self._cached_alpha: Optional[np.ndarray] = None
        self._cached_x: int = 0
        self._cached_y: int = 0
        self._cached_w: int = 0
        self._cached_h: int = 0
    
    def _prepare_for_frame_size(self, frame_h: int, frame_w: int):
        """Pre-compute the scaled logo and position for a given frame size."""
        frame_size = (frame_h, frame_w)
        if self._cached_frame_size == frame_size:
            return
        
        # Scale logo to target width
        logo_w = max(int(frame_w * self._scale), 1)
        aspect = self._logo_rgba.shape[0] / self._logo_rgba.shape[1]
        logo_h = max(int(logo_w * aspect), 1)
        
        scaled = cv2.resize(self._logo_rgba, (logo_w, logo_h), interpolation=cv2.INTER_AREA)
        
        # Split RGB and alpha
        self._cached_logo_rgb = scaled[:, :, :3].astype(np.float32)
        alpha = scaled[:, :, 3].astype(np.float32) / 255.0 * self._opacity
        # Expand alpha to 3 channels
        self._cached_alpha = np.stack([alpha] * 3, axis=2)
        
        # Compute position
        m = self._margin
        if self._position == "bottom-right":
            self._cached_x = frame_w - logo_w - m
            self._cached_y = frame_h - logo_h - m
        elif self._position == "bottom-left":
            self._cached_x = m
            self._cached_y = frame_h - logo_h - m
        elif self._position == "top-right":
            self._cached_x = frame_w - logo_w - m
            self._cached_y = m
        elif self._position == "top-left":
            self._cached_x = m
            self._cached_y = m
        else:
            raise ValueError(f"Unknown position: {self._position}")
        
        # Clamp to valid range
        self._cached_x = max(0, self._cached_x)
        self._cached_y = max(0, self._cached_y)
        self._cached_w = min(logo_w, frame_w - self._cached_x)
        self._cached_h = min(logo_h, frame_h - self._cached_y)
        
        # Trim cached arrays if they were clamped
        self._cached_logo_rgb = self._cached_logo_rgb[:self._cached_h, :self._cached_w]
        self._cached_alpha = self._cached_alpha[:self._cached_h, :self._cached_w]
        
        self._cached_frame_size = frame_size
    
    def apply(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Overlay watermark onto a frame.
        
        Args:
            frame_rgb: RGB frame as numpy array (H, W, 3), dtype uint8
            
        Returns:
            Frame with watermark applied (same shape and dtype)
        """
        h, w = frame_rgb.shape[:2]
        self._prepare_for_frame_size(h, w)
        
        if self._cached_w <= 0 or self._cached_h <= 0:
            return frame_rgb
        
        # Extract the ROI
        x, y = self._cached_x, self._cached_y
        roi_w, roi_h = self._cached_w, self._cached_h
        
        roi = frame_rgb[y:y + roi_h, x:x + roi_w].astype(np.float32)
        
        # Alpha blend: result = alpha * logo + (1 - alpha) * background
        blended = (
            self._cached_alpha * self._cached_logo_rgb
            + (1.0 - self._cached_alpha) * roi
        )
        
        # Write back
        result = frame_rgb.copy()
        result[y:y + roi_h, x:x + roi_w] = np.clip(blended, 0, 255).astype(np.uint8)
        
        return result
