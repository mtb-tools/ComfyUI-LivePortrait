"""
Patch the live portrait project methods to make it Comfy friendly
"""

from pathlib import Path
import comfy.utils
import folder_paths
import cv2
import torch

from .liveportrait.src.utils.crop import _transform_img

from .liveportrait.src.live_portrait_wrapper import (
    LivePortraitWrapper,
    calc_lip_close_ratio,
)
from .liveportrait.src.utils.io import resize_to_limit
from .liveportrait.src.utils.camera import get_rotation_matrix
from .liveportrait.src.config.crop_config import CropConfig
from .liveportrait.src.utils.cropper import Cropper
from .liveportrait.src.utils.landmark_runner import LandmarkRunner
from .liveportrait.src.utils.face_analysis_diy import FaceAnalysisDIY
from .liveportrait.src.live_portrait_pipeline import LivePortraitPipeline
from .liveportrait.src.config.inference_config import InferenceConfig
from .liveportrait.src.utils.timer import Timer
import numpy as np


def apply_config(base, **kwargs):
    for k, v in kwargs.items():
        setattr(base, k, v)


def wrapper_init(self: LivePortraitWrapper, cfg: InferenceConfig):
    """
    The original init loads ckpt from path, we load safetensors and set the fields from the pipeline
    """
    self.cfg = cfg
    self.device_id = cfg.device_id
    self.timer = Timer()


def pipeline_init(
    self: LivePortraitPipeline,
    inference_cfg: InferenceConfig,
    crop_cfg: CropConfig,
    safetensors,
):
    self.live_portrait_wrapper = LivePortraitWrapper(cfg=inference_cfg)

    apply_config(self.live_portrait_wrapper, **safetensors)
    self.cropper = Cropper(crop_cfg=crop_cfg)


def cropper_init(self: Cropper, **kwargs) -> None:
    device_id = kwargs.get("device_id", 0)
    models_dir = Path(folder_paths.models_dir)
    self.landmark_runner = LandmarkRunner(
        # ckpt_path=make_abs_path('../../pretrained_weights/liveportrait/landmark.onnx'),
        ckpt_path=(models_dir / "liveportrait" / "landmark.onnx").as_posix(),
        onnx_provider="cuda",
        device_id=device_id,
    )
    self.landmark_runner.warmup()

    self.face_analysis_wrapper = FaceAnalysisDIY(
        name="buffalo_l",
        root=(models_dir / "insightface").as_posix(),
        providers=["CUDAExecutionProvider"],
    )
    self.face_analysis_wrapper.prepare(ctx_id=device_id, det_size=(512, 512))
    self.face_analysis_wrapper.warmup()

    self.crop_cfg = kwargs.get("crop_cfg", None)


def pipeline_execute(
    self: LivePortraitPipeline, img_rgb, driving_images_np=None, driving_template=None
):
    inference_cfg = self.live_portrait_wrapper.cfg  # for convenience
    ######## process reference portrait ########
    img_rgb = resize_to_limit(
        img_rgb, inference_cfg.ref_max_shape, inference_cfg.ref_shape_n
    )
    crop_info = self.cropper.crop_single_image(img_rgb)
    source_lmk = crop_info["lmk_crop"]
    img_crop, img_crop_256x256 = crop_info["img_crop"], crop_info["img_crop_256x256"]
    if inference_cfg.flag_do_crop:
        I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
    else:
        I_s = self.live_portrait_wrapper.prepare_source(img_rgb)
    x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
    x_c_s = x_s_info["kp"]
    R_s = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
    f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
    x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

    if inference_cfg.flag_lip_zero:
        # let lip-open scalar to be 0 at first
        c_d_lip_before_animation = [0.0]
        combined_lip_ratio_tensor_before_animation = (
            self.live_portrait_wrapper.calc_combined_lip_ratio(
                c_d_lip_before_animation, source_lmk
            )
        )
        if (
            combined_lip_ratio_tensor_before_animation[0][0]
            < inference_cfg.lip_zero_threshold
        ):
            inference_cfg.flag_lip_zero = False
        else:
            lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(
                x_s, combined_lip_ratio_tensor_before_animation
            )
    ############################################

    ######## process driving info ########

    driving_rgb_lst = driving_images_np

    driving_rgb_lst_256 = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
    I_d_lst = self.live_portrait_wrapper.prepare_driving_videos(driving_rgb_lst_256)
    n_frames = I_d_lst.shape[0]
    if inference_cfg.flag_eye_retargeting or inference_cfg.flag_lip_retargeting:
        driving_lmk_lst = self.cropper.get_retargeting_lmk_info(driving_rgb_lst)
        input_eye_ratio_lst, input_lip_ratio_lst = (
            self.live_portrait_wrapper.calc_retargeting_ratio(
                source_lmk, driving_lmk_lst
            )
        )


    ######## prepare for pasteback ########
    if inference_cfg.flag_pasteback:
        if inference_cfg.mask_crop is None:
            inference_cfg.mask_crop = cv2.imread(
                (
                    Path(__file__).parent
                    / "liveportrait"
                    / "src"
                    / "utils"
                    / "resources"
                    / "mask_template.png"
                )
                .resolve()
                .as_posix(),
                cv2.IMREAD_COLOR,
            )
        mask_ori = _transform_img(
            inference_cfg.mask_crop,
            crop_info["M_c2o"],
            dsize=(img_rgb.shape[1], img_rgb.shape[0]),
        )
        mask_ori = mask_ori.astype(np.float32) / 255.0
        I_p_paste_lst = []
    #########################################

    I_p_lst = []
    R_d_0, x_d_0_info = None, None
    pbar = comfy.utils.ProgressBar(n_frames)
    for i in range(n_frames):
        I_d_i = I_d_lst[i]
        x_d_i_info = self.live_portrait_wrapper.get_kp_info(I_d_i)
        R_d_i = get_rotation_matrix(
            x_d_i_info["pitch"], x_d_i_info["yaw"], x_d_i_info["roll"]
        )
        if i == 0:
            R_d_0 = R_d_i
            x_d_0_info = x_d_i_info

        if inference_cfg.flag_relative:
            R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
            delta_new = x_s_info["exp"] + (x_d_i_info["exp"] - x_d_0_info["exp"])
            scale_new = x_s_info["scale"] * (x_d_i_info["scale"] / x_d_0_info["scale"])
            t_new = x_s_info["t"] + (x_d_i_info["t"] - x_d_0_info["t"])
        else:
            R_new = R_d_i
            delta_new = x_d_i_info["exp"]
            scale_new = x_s_info["scale"]
            t_new = x_d_i_info["t"]

        t_new[..., 2].fill_(0)  # zero tz
        x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

        # Algorithm 1:
        if (
            not inference_cfg.flag_stitching
            and not inference_cfg.flag_eye_retargeting
            and not inference_cfg.flag_lip_retargeting
        ):
            # without stitching or retargeting
            if inference_cfg.flag_lip_zero:
                x_d_i_new += lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
            else:
                pass
        elif (
            inference_cfg.flag_stitching
            and not inference_cfg.flag_eye_retargeting
            and not inference_cfg.flag_lip_retargeting
        ):
            # with stitching and without retargeting
            if inference_cfg.flag_lip_zero:
                x_d_i_new = self.live_portrait_wrapper.stitching(
                    x_s, x_d_i_new
                ) + lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
            else:
                x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
        else:
            eyes_delta, lip_delta = None, None
            if inference_cfg.flag_eye_retargeting:
                c_d_eyes_i = input_eye_ratio_lst[i]
                combined_eye_ratio_tensor = (
                    self.live_portrait_wrapper.calc_combined_eye_ratio(
                        c_d_eyes_i, source_lmk
                    )
                )
                combined_eye_ratio_tensor = (
                    combined_eye_ratio_tensor
                    * inference_cfg.eyes_retargeting_multiplier
                )
                # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                eyes_delta = self.live_portrait_wrapper.retarget_eye(
                    x_s, combined_eye_ratio_tensor
                )
            if inference_cfg.flag_lip_retargeting:
                c_d_lip_i = input_lip_ratio_lst[i]
                combined_lip_ratio_tensor = (
                    self.live_portrait_wrapper.calc_combined_lip_ratio(
                        c_d_lip_i, source_lmk
                    )
                )
                combined_lip_ratio_tensor = (
                    combined_lip_ratio_tensor * inference_cfg.lip_retargeting_multiplier
                )
                # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                lip_delta = self.live_portrait_wrapper.retarget_lip(
                    x_s, combined_lip_ratio_tensor
                )

            if inference_cfg.flag_relative:  # use x_s
                x_d_i_new = (
                    x_s
                    + (
                        eyes_delta.reshape(-1, x_s.shape[1], 3)
                        if eyes_delta is not None
                        else 0
                    )
                    + (
                        lip_delta.reshape(-1, x_s.shape[1], 3)
                        if lip_delta is not None
                        else 0
                    )
                )
            else:  # use x_d,i
                x_d_i_new = (
                    x_d_i_new
                    + (
                        eyes_delta.reshape(-1, x_s.shape[1], 3)
                        if eyes_delta is not None
                        else 0
                    )
                    + (
                        lip_delta.reshape(-1, x_s.shape[1], 3)
                        if lip_delta is not None
                        else 0
                    )
                )

            if inference_cfg.flag_stitching:
                x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

        out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
        I_p_i = self.live_portrait_wrapper.parse_output(out["out"])[0]
        I_p_lst.append(I_p_i)
        pbar.update(1)

        # if inference_cfg.flag_pasteback:
        I_p_i_to_ori = _transform_img(
            I_p_i, crop_info["M_c2o"], dsize=(img_rgb.shape[1], img_rgb.shape[0])
        )
        I_p_i_to_ori_blend = np.clip(
            mask_ori * I_p_i_to_ori + (1 - mask_ori) * img_rgb, 0, 255
        ).astype(np.uint8)
        out = np.hstack([I_p_i_to_ori, I_p_i_to_ori_blend])
        I_p_paste_lst.append(I_p_i_to_ori_blend)

    return I_p_lst, I_p_paste_lst


def calc_combined_lip_ratio(self, input_lip_ratio, source_lmk):
    lip_close_ratio = calc_lip_close_ratio(source_lmk[None])
    lip_close_ratio_tensor = (
        torch.from_numpy(lip_close_ratio).float().cuda(self.device_id)
    )
    # [c_s,lip, c_d,lip,i]
    input_lip_ratio_tensor = torch.Tensor(np.array([input_lip_ratio[0]])).cuda(
        self.device_id
    )
    if input_lip_ratio_tensor.shape != [1, 1]:
        input_lip_ratio_tensor = input_lip_ratio_tensor.reshape(1, 1)
    combined_lip_ratio_tensor = torch.cat(
        [lip_close_ratio_tensor, input_lip_ratio_tensor], dim=1
    )
    return combined_lip_ratio_tensor


LivePortraitWrapper.__init__ = wrapper_init
LivePortraitPipeline.__init__ = pipeline_init
LivePortraitPipeline.execute = pipeline_execute
Cropper.__init__ = cropper_init

__all__ = ["LivePortraitWrapper", "LivePortraitPipeline", "Cropper"]
