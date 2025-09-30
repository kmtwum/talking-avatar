import threading
import queue
import numpy as np
import traceback
from tqdm import tqdm
import torch
import gc

from core.atomic_components.avatar_registrar import AvatarRegistrar, smooth_x_s_info_lst
from core.atomic_components.condition_handler import ConditionHandler, _mirror_index
from core.atomic_components.audio2motion import Audio2Motion
from core.atomic_components.motion_stitch import MotionStitch
from core.atomic_components.warp_f3d import WarpF3D
from core.atomic_components.decode_f3d import DecodeF3D
from core.atomic_components.putback import PutBack
from core.atomic_components.writer import VideoWriterByImageIO
from core.atomic_components.wav2feat import Wav2Feat
from core.atomic_components.cfg import parse_cfg, print_cfg


class FastStreamSDK:
    """Optimized version of StreamSDK for better performance"""
    
    def __init__(self, cfg_pkl, data_root, **kwargs):
        # Enable memory optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Parse config with optimizations
        [
            avatar_registrar_cfg,
            condition_handler_cfg,
            lmdm_cfg,
            stitch_network_cfg,
            warp_network_cfg,
            decoder_cfg,
            wav2feat_cfg,
            default_kwargs,
        ] = parse_cfg(cfg_pkl, data_root, kwargs)
        
        self.default_kwargs = default_kwargs
        
        # Initialize components
        self.avatar_registrar = AvatarRegistrar(**avatar_registrar_cfg)
        self.condition_handler = ConditionHandler(**condition_handler_cfg)
        self.audio2motion = Audio2Motion(lmdm_cfg)
        self.motion_stitch = MotionStitch(stitch_network_cfg)
        self.warp_f3d = WarpF3D(warp_network_cfg)
        self.decode_f3d = DecodeF3D(decoder_cfg)
        self.putback = PutBack()
        self.wav2feat = Wav2Feat(**wav2feat_cfg)

    def setup(self, source_path, output_path, **kwargs):
        # Merge with optimized defaults
        kwargs = self._merge_kwargs(self.default_kwargs, kwargs)
        
        # Apply performance optimizations
        kwargs.update({
            "sampling_timesteps": kwargs.get("sampling_timesteps", 25),  # Reduce from 50
            "max_size": kwargs.get("max_size", 512),  # Reduce resolution
            "smo_k_s": kwargs.get("smo_k_s", 7),  # Reduce smoothing
            "smo_k_d": kwargs.get("smo_k_d", 1),  # Reduce smoothing
        })
        
        print("=== Fast Mode Optimizations ===")
        print_cfg(**kwargs)
        print("=" * 32)
        
        # Setup components with optimized parameters
        self._setup_components(source_path, output_path, **kwargs)
        
        # Setup optimized worker threads
        self._setup_workers()

    def _setup_components(self, source_path, output_path, **kwargs):
        """Setup all components with optimized parameters"""
        
        # Extract parameters
        self.max_size = kwargs.get("max_size", 512)
        self.template_n_frames = kwargs.get("template_n_frames", -1)
        self.crop_scale = kwargs.get("crop_scale", 2.0)
        self.crop_vx_ratio = kwargs.get("crop_vx_ratio", 0)
        self.crop_vy_ratio = kwargs.get("crop_vy_ratio", -0.125)
        self.crop_flag_do_rot = kwargs.get("crop_flag_do_rot", True)
        self.smo_k_s = kwargs.get('smo_k_s', 7)
        
        # Condition handler params
        self.emo = kwargs.get("emo", 4)
        self.eye_f0_mode = kwargs.get("eye_f0_mode", False)
        self.ch_info = kwargs.get("ch_info", None)
        
        # Audio2motion params
        self.overlap_v2 = kwargs.get("overlap_v2", 10)
        self.fix_kp_cond = kwargs.get("fix_kp_cond", 0)
        self.fix_kp_cond_dim = kwargs.get("fix_kp_cond_dim", None)
        self.sampling_timesteps = kwargs.get("sampling_timesteps", 25)
        self.online_mode = kwargs.get("online_mode", False)
        self.v_min_max_for_clip = kwargs.get('v_min_max_for_clip', None)
        self.smo_k_d = kwargs.get("smo_k_d", 1)
        
        # Motion stitch params
        self.N_d = kwargs.get("N_d", -1)
        self.use_d_keys = kwargs.get("use_d_keys", None)
        self.relative_d = kwargs.get("relative_d", True)
        self.drive_eye = kwargs.get("drive_eye", None)
        self.delta_eye_arr = kwargs.get("delta_eye_arr", None)
        self.delta_eye_open_n = kwargs.get("delta_eye_open_n", 0)
        self.fade_type = kwargs.get("fade_type", "")
        self.fade_out_keys = kwargs.get("fade_out_keys", ("exp",))
        self.flag_stitching = kwargs.get("flag_stitching", True)
        
        self.ctrl_info = kwargs.get("ctrl_info", dict())
        self.overall_ctrl_info = kwargs.get("overall_ctrl_info", dict())
        
        # Register avatar with optimized crop
        crop_kwargs = {
            "crop_scale": self.crop_scale,
            "crop_vx_ratio": self.crop_vx_ratio,
            "crop_vy_ratio": self.crop_vy_ratio,
            "crop_flag_do_rot": self.crop_flag_do_rot,
        }
        
        n_frames = self.template_n_frames if self.template_n_frames > 0 else self.N_d
        source_info = self.avatar_registrar(
            source_path, 
            max_dim=self.max_size, 
            n_frames=n_frames, 
            **crop_kwargs,
        )
        
        # Apply smoothing if needed
        if len(source_info["x_s_info_lst"]) > 1 and self.smo_k_s > 1:
            source_info["x_s_info_lst"] = smooth_x_s_info_lst(
                source_info["x_s_info_lst"], smo_k=self.smo_k_s
            )
        
        self.source_info = source_info
        self.source_info_frames = len(source_info["x_s_info_lst"])
        
        # Setup condition handler
        self.condition_handler.setup(
            source_info, self.emo, eye_f0_mode=self.eye_f0_mode, ch_info=self.ch_info
        )
        
        # Setup audio2motion
        x_s_info_0 = self.condition_handler.x_s_info_0
        self.audio2motion.setup(
            x_s_info_0, 
            overlap_v2=self.overlap_v2,
            fix_kp_cond=self.fix_kp_cond,
            fix_kp_cond_dim=self.fix_kp_cond_dim,
            sampling_timesteps=self.sampling_timesteps,
            online_mode=self.online_mode,
            v_min_max_for_clip=self.v_min_max_for_clip,
            smo_k_d=self.smo_k_d,
        )
        
        # Setup motion stitch
        is_image_flag = source_info["is_image_flag"]
        x_s_info = source_info['x_s_info_lst'][0]
        self.motion_stitch.setup(
            N_d=self.N_d,
            use_d_keys=self.use_d_keys,
            relative_d=self.relative_d,
            drive_eye=self.drive_eye,
            delta_eye_arr=self.delta_eye_arr,
            delta_eye_open_n=self.delta_eye_open_n,
            fade_out_keys=self.fade_out_keys,
            fade_type=self.fade_type,
            flag_stitching=self.flag_stitching,
            is_image_flag=is_image_flag,
            x_s_info=x_s_info,
            d0=None,
            ch_info=self.ch_info,
            overall_ctrl_info=self.overall_ctrl_info,
        )
        
        # Setup video writer
        self.output_path = output_path
        self.tmp_output_path = output_path + ".tmp.mp4"
        self.writer = VideoWriterByImageIO(self.tmp_output_path)
        self.writer_pbar = tqdm(desc="writer")
        
        # Setup audio buffer
        if self.online_mode:
            self.audio_feat = self.wav2feat.wav2feat(
                np.zeros((self.overlap_v2 * 640,), dtype=np.float32), sr=16000
            )
            assert len(self.audio_feat) == self.overlap_v2
        else:
            self.audio_feat = np.zeros((0, self.wav2feat.feat_dim), dtype=np.float32)
        self.cond_idx_start = 0 - len(self.audio_feat)

    def _setup_workers(self):
        """Setup optimized worker threads with larger queues"""
        QUEUE_MAX_SIZE = 200  # Increased from 100
        
        self.worker_exception = None
        self.stop_event = threading.Event()
        
        # Create queues
        self.audio2motion_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.motion_stitch_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.warp_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.decode_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.putback_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.writer_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        
        # Create worker threads
        self.thread_list = [
            threading.Thread(target=self.audio2motion_worker, daemon=True),
            threading.Thread(target=self.motion_stitch_worker, daemon=True),
            threading.Thread(target=self.warp_f3d_worker, daemon=True),
            threading.Thread(target=self.decode_f3d_worker, daemon=True),
            threading.Thread(target=self.putback_worker, daemon=True),
            threading.Thread(target=self.writer_worker, daemon=True),
        ]
        
        # Start all threads
        for thread in self.thread_list:
            thread.start()

    def _merge_kwargs(self, default_kwargs, run_kwargs):
        """Merge kwargs with defaults"""
        for k, v in default_kwargs.items():
            if k not in run_kwargs:
                run_kwargs[k] = v
        return run_kwargs

    def setup_Nd(self, N_d, fade_in=-1, fade_out=-1, ctrl_info=None):
        """Setup number of frames and fade effects"""
        self.motion_stitch.set_Nd(N_d)
        
        if ctrl_info is None:
            ctrl_info = self.ctrl_info
            
        # Apply fade effects
        if fade_in > 0:
            for i in range(fade_in):
                alpha = i / fade_in
                item = ctrl_info.get(i, {})
                item["fade_alpha"] = alpha
                ctrl_info[i] = item
                
        if fade_out > 0:
            ss = N_d - fade_out - 1
            ee = N_d - 1
            for i in range(ss, N_d):
                alpha = max((ee - i) / (ee - ss), 0)
                item = ctrl_info.get(i, {})
                item["fade_alpha"] = alpha
                ctrl_info[i] = item
                
        self.ctrl_info = ctrl_info

    def _get_ctrl_info(self, fid):
        """Get control info for frame"""
        try:
            if isinstance(self.ctrl_info, dict):
                return self.ctrl_info.get(fid, {})
            elif isinstance(self.ctrl_info, list):
                return self.ctrl_info[fid]
            else:
                return {}
        except Exception:
            return {}

    # Worker methods (simplified versions of original)
    def audio2motion_worker(self):
        """Optimized audio2motion worker"""
        try:
            self._audio2motion_offline_fast()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _audio2motion_offline_fast(self):
        """Fast offline audio2motion processing"""
        while not self.stop_event.is_set():
            try:
                item = self.audio2motion_queue.get(timeout=1)
            except queue.Empty:
                continue
                
            if item is None:
                break
                
            aud_feat = item
            
            # Process audio with optimized batching
            aud_cond_all = self.condition_handler(aud_feat, 0)
            seq_frames = self.audio2motion.seq_frames
            valid_clip_len = self.audio2motion.valid_clip_len
            num_frames = len(aud_cond_all)
            
            # Process in larger batches for efficiency
            batch_size = min(seq_frames * 2, num_frames)
            idx = 0
            res_kp_seq = None
            
            pbar = tqdm(desc="audio2motion", total=num_frames)
            while idx < num_frames:
                end_idx = min(idx + batch_size, num_frames)
                aud_cond = aud_cond_all[idx:end_idx][None]
                
                if aud_cond.shape[1] < seq_frames:
                    pad = np.stack([aud_cond[:, -1]] * (seq_frames - aud_cond.shape[1]), 1)
                    aud_cond = np.concatenate([aud_cond, pad], 1)
                    
                res_kp_seq = self.audio2motion(aud_cond, res_kp_seq)
                pbar.update(min(valid_clip_len, end_idx - idx))
                idx += valid_clip_len
                
                # Clear GPU cache periodically
                if idx % (seq_frames * 4) == 0:
                    torch.cuda.empty_cache()
                    
            pbar.close()
            
            # Apply smoothing and convert format
            res_kp_seq = res_kp_seq[:, :num_frames]
            res_kp_seq = self.audio2motion._smo(res_kp_seq, 0, res_kp_seq.shape[1])
            x_d_info_list = self.audio2motion.cvt_fmt(res_kp_seq)
            
            # Queue results
            gen_frame_idx = 0
            for x_d_info in x_d_info_list:
                frame_idx = _mirror_index(gen_frame_idx, self.source_info_frames)
                ctrl_kwargs = self._get_ctrl_info(gen_frame_idx)
                
                while not self.stop_event.is_set():
                    try:
                        self.motion_stitch_queue.put([frame_idx, x_d_info, ctrl_kwargs], timeout=1)
                        break
                    except queue.Full:
                        continue
                gen_frame_idx += 1
                
            break
            
        self.motion_stitch_queue.put(None)

    # Other worker methods (motion_stitch, warp_f3d, decode_f3d, putback, writer)
    # These remain largely the same as the original but with optimized queue handling
    
    def motion_stitch_worker(self):
        try:
            self._motion_stitch_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _motion_stitch_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.motion_stitch_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.warp_f3d_queue.put(None)
                break
            
            frame_idx, x_d_info, ctrl_kwargs = item
            x_s_info = self.source_info["x_s_info_lst"][frame_idx]
            x_s, x_d = self.motion_stitch(x_s_info, x_d_info, **ctrl_kwargs)
            self.warp_f3d_queue.put([frame_idx, x_s, x_d])

    def warp_f3d_worker(self):
        try:
            self._warp_f3d_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _warp_f3d_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.warp_f3d_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.decode_f3d_queue.put(None)
                break
            frame_idx, x_s, x_d = item
            f_s = self.source_info["f_s_lst"][frame_idx]
            f_3d = self.warp_f3d(f_s, x_s, x_d)
            self.decode_f3d_queue.put([frame_idx, f_3d])

    def decode_f3d_worker(self):
        try:
            self._decode_f3d_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _decode_f3d_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.decode_f3d_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.putback_queue.put(None)
                break
            frame_idx, f_3d = item
            render_img = self.decode_f3d(f_3d)
            self.putback_queue.put([frame_idx, render_img])

    def putback_worker(self):
        try:
            self._putback_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _putback_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.putback_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.writer_queue.put(None)
                break
            frame_idx, render_img = item
            frame_rgb = self.source_info["img_rgb_lst"][frame_idx]
            M_c2o = self.source_info["M_c2o_lst"][frame_idx]
            res_frame_rgb = self.putback(frame_rgb, render_img, M_c2o)
            self.writer_queue.put(res_frame_rgb)

    def writer_worker(self):
        try:
            self._writer_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _writer_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.writer_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                break
            res_frame_rgb = item
            self.writer(res_frame_rgb, fmt="rgb")
            self.writer_pbar.update()

    def close(self):
        """Close SDK and cleanup resources"""
        # Signal end of processing
        self.audio2motion_queue.put(None)
        
        # Wait for all threads to complete
        for thread in self.thread_list:
            thread.join(timeout=30)  # Add timeout to prevent hanging
            
        # Close writer
        try:
            self.writer.close()
            self.writer_pbar.close()
        except:
            traceback.print_exc()
            
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Check for worker exceptions
        if self.worker_exception is not None:
            raise self.worker_exception

    def run_chunk(self, audio_chunk, chunksize=(3, 5, 2)):
        """Process audio chunk (for online mode)"""
        aud_feat = self.wav2feat(audio_chunk, chunksize=chunksize)
        while not self.stop_event.is_set():
            try:
                self.audio2motion_queue.put(aud_feat, timeout=1)
                break
            except queue.Full:
                continue