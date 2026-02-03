import cv2
import numpy as np
from ..utils.blend import blend_images_cy
from ..utils.get_mask import get_mask


def match_color_histogram(source, reference, mask=None):
    """
    Match the color histogram of the source image to the reference image.
    This helps preserve the original image colors in the rendered output.
    
    Args:
        source: The rendered image (RGB, float or uint8)
        reference: The original source image (RGB, uint8)
        mask: Optional mask for the face region (0-1 float)
    
    Returns:
        Color-matched image
    """
    # Convert to float if needed
    if source.dtype == np.uint8:
        source = source.astype(np.float32)
    if reference.dtype == np.uint8:
        reference = reference.astype(np.float32)
    
    # Use LAB color space for better perceptual color matching
    source_lab = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    reference_lab = cv2.cvtColor(reference.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Calculate mean and std for each channel
    result_lab = source_lab.copy()
    
    for i in range(3):
        # Get source and reference stats
        src_mean = source_lab[:, :, i].mean()
        src_std = source_lab[:, :, i].std() + 1e-6
        ref_mean = reference_lab[:, :, i].mean()
        ref_std = reference_lab[:, :, i].std() + 1e-6
        
        # Apply color transfer: normalize, scale, and shift
        result_lab[:, :, i] = (source_lab[:, :, i] - src_mean) * (ref_std / src_std) + ref_mean
    
    # Clip to valid range
    result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 255)  # L channel
    result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], 0, 255)  # A channel
    result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], 0, 255)  # B channel
    
    # Convert back to RGB
    result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    return result.astype(np.float32)


class PutBackNumpy:
    def __init__(
        self,
        mask_template_path=None,
        color_match=True,
    ):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            self.mask_ori_float = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR)
            self.mask_ori_float = mask.astype(np.float32) / 255.0
        self.color_match = color_match
        self._reference_crop = None  # Will store the source crop for color matching

    def set_reference_crop(self, crop_rgb):
        """Set the reference crop for color matching (512x512 source crop)."""
        self._reference_crop = crop_rgb.copy() if crop_rgb is not None else None

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        
        # Apply color matching to preserve original colors
        if self.color_match and self._reference_crop is not None:
            # Resize reference to match render_image size if needed
            rh, rw = render_image.shape[:2]
            ref_resized = cv2.resize(self._reference_crop, (rw, rh), interpolation=cv2.INTER_AREA)
            render_image = match_color_histogram(render_image, ref_resized)
        
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        result = mask_warped * frame_warped + (1 - mask_warped) * frame_rgb
        result = np.clip(result, 0, 255)
        result = result.astype(np.uint8)
        return result
    

class PutBack:
    def __init__(
        self,
        mask_template_path=None,
        color_match=True,
    ):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            mask = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        self.mask_ori_float = np.ascontiguousarray(mask)[:,:,0]
        self.result_buffer = None
        self.color_match = color_match
        self._reference_crop = None  # Will store the source crop for color matching

    def set_reference_crop(self, crop_rgb):
        """Set the reference crop for color matching (512x512 source crop)."""
        self._reference_crop = crop_rgb.copy() if crop_rgb is not None else None

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        
        # Apply color matching to preserve original colors
        if self.color_match and self._reference_crop is not None:
            # Resize reference to match render_image size if needed
            rh, rw = render_image.shape[:2]
            ref_resized = cv2.resize(self._reference_crop, (rw, rh), interpolation=cv2.INTER_AREA)
            render_image = match_color_histogram(render_image, ref_resized)
        
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        self.result_buffer = np.empty((h, w, 3), dtype=np.uint8)

        # Use Cython implementation for blending
        blend_images_cy(mask_warped, frame_warped, frame_rgb, self.result_buffer)

        return self.result_buffer