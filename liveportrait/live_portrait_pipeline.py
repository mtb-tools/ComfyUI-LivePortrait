# coding: utf-8

"""
Pipeline of LivePortrait
"""

import cv2
import comfy.utils

import os.path as osp
import numpy as np
from .config.inference_config import InferenceConfig

from .utils.camera import get_rotation_matrix
from .utils.crop import _transform_img
from .live_portrait_wrapper import LivePortraitWrapper
from ..log import logger


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class LivePortraitPipeline(object):
    def __init__(
        self,
        *,
        appearance_feature_extractor,
        motion_extractor,
        warping_module,
        spade_generator,
        stitching_retargeting_module,
        inference_cfg: InferenceConfig,
    ):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(
            appearance_feature_extractor,
            motion_extractor,
            warping_module,
            spade_generator,
            stitching_retargeting_module,
            cfg=inference_cfg,
        )

    def _get_source_frame(
        self, source_np: np.ndarray, idx: int, total_frames: int, method: str
    ):
        if source_np.shape[0] == 1:
            return source_np[0]

        if method == "repeat":
            return source_np[min(idx, source_np.shape[0] - 1)]
        elif method == "cycle":
            return source_np[idx % source_np.shape[0]]
        elif method == "mirror":
            cycle_length = 2 * source_np.shape[0] - 2
            mirror_idx = idx % cycle_length
            if mirror_idx >= source_np.shape[0]:
                mirror_idx = cycle_length - mirror_idx
            return source_np[mirror_idx]
        elif method == "nearest":
            ratio = idx / (total_frames - 1)
            return source_np[
                min(int(ratio * (source_np.shape[0] - 1)), source_np.shape[0] - 1)
            ]

    def execute(
        self,
        source_images: np.ndarray,
        driving_images: np.ndarray,
        mismatch_method: str = "repeat",
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        config = self.live_portrait_wrapper.cfg
        total_frames = driving_images.shape[0]

        cropped_outputs = []
        full_outputs = []

        progress_bar = comfy.utils.ProgressBar(total_frames)

        driving_landmarks = self._get_driving_landmarks(driving_images, config)

        for frame_index in range(total_frames):
            try:
                source_frame = self._get_source_frame(
                    source_images, frame_index, total_frames, mismatch_method
                )
                driving_frame = driving_images[frame_index]

                cropped_output, full_output = self._process_frame(
                    source_frame, driving_frame, frame_index, config, driving_landmarks
                )

                cropped_outputs.append(cropped_output)
                full_outputs.append(full_output)

                progress_bar.update(1)
            except Exception as e:
                logger.error(f"Error processing frame {frame_index}: {str(e)}")
                raise

        return cropped_outputs, full_outputs

    def _get_driving_landmarks(self, driving_images: np.ndarray, config) -> list:
        if config.flag_eye_retargeting or config.flag_lip_retargeting:
            return self.cropper.get_retargeting_lmk_info(driving_images)
        return []

    def _process_frame(
        self,
        source_frame: np.ndarray,
        driving_frame: np.ndarray,
        frame_index: int,
        config,
        driving_landmarks: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        crop_info = self.cropper.crop_single_image(source_frame)
        source_landmarks = crop_info["lmk_crop"]
        cropped_source = (
            crop_info["img_crop_256x256"] if config.flag_do_crop else source_frame
        )

        source_features = self._extract_source_features(cropped_source)
        driving_features = self._extract_driving_features(driving_frame)

        keypoints = self._compute_keypoints(
            source_features, driving_features, frame_index, config
        )

        if config.flag_eye_retargeting or config.flag_lip_retargeting:
            keypoints = self._apply_retargeting(
                keypoints, source_landmarks, driving_landmarks, frame_index, config
            )

        warped_output = self._warp_and_decode(source_features, keypoints)

        cropped_output = self.live_portrait_wrapper.parse_output(warped_output["out"])[
            0
        ]
        full_output = self._transform_and_blend(
            cropped_output, source_frame, crop_info, config
        )

        return cropped_output, full_output

    def _extract_source_features(self, source_image: np.ndarray):
        prepared_source = self.live_portrait_wrapper.prepare_source(source_image)
        kp_src_info = self.live_portrait_wrapper.get_kp_info(prepared_source)
        feature_3d = self.live_portrait_wrapper.extract_feature_3d(prepared_source)
        return {
            "prepared": prepared_source,
            "keypoints": kp_src_info,
            "feature_3d": feature_3d,
            "transformed_kp": self.live_portrait_wrapper.transform_keypoint(
                kp_src_info
            ),
        }

    def _extract_driving_features(self, driving_frame: np.ndarray):
        driving_frame_resized = cv2.resize(driving_frame, (256, 256))
        prepared_driving = self.live_portrait_wrapper.prepare_driving_videos(
            [driving_frame_resized]
        )[0]
        return self.live_portrait_wrapper.get_kp_info(prepared_driving)

    def _compute_keypoints(
        self, source_features, driving_features, frame_index: int, config
    ):
        if frame_index == 0:
            self.initial_rotation = get_rotation_matrix(
                driving_features["pitch"],
                driving_features["yaw"],
                driving_features["roll"],
            )
            self.initial_keypoints = driving_features

        if config.flag_relative:
            return self._compute_relative_keypoints(source_features, driving_features)
        return self._compute_absolute_keypoints(source_features, driving_features)

    def _compute_relative_keypoints(self, source_features, driving_features):
        R_new = (
            get_rotation_matrix(
                driving_features["pitch"],
                driving_features["yaw"],
                driving_features["roll"],
            )
            @ self.initial_rotation.permute(0, 2, 1)
        ) @ get_rotation_matrix(
            source_features["keypoints"]["pitch"],
            source_features["keypoints"]["yaw"],
            source_features["keypoints"]["roll"],
        )
        delta_new = source_features["keypoints"]["exp"] + (
            driving_features["exp"] - self.initial_keypoints["exp"]
        )
        scale_new = source_features["keypoints"]["scale"] * (
            driving_features["scale"] / self.initial_keypoints["scale"]
        )
        t_new = source_features["keypoints"]["t"] + (
            driving_features["t"] - self.initial_keypoints["t"]
        )
        t_new[..., 2].fill_(0)  # zero tz
        return (
            scale_new * (source_features["keypoints"]["kp"] @ R_new + delta_new) + t_new
        )

    def _compute_absolute_keypoints(self, source_features, driving_features):
        R_new = get_rotation_matrix(
            driving_features["pitch"], driving_features["yaw"], driving_features["roll"]
        )
        t_new = driving_features["t"]
        t_new[..., 2].fill_(0)  # zero tz
        return (
            source_features["keypoints"]["scale"]
            * (source_features["keypoints"]["kp"] @ R_new + driving_features["exp"])
            + t_new
        )

    def _apply_retargeting(
        self, keypoints, source_landmarks, driving_landmarks, frame_index: int, config
    ):
        if config.flag_eye_retargeting:
            keypoints = self._apply_eye_retargeting(
                keypoints, source_landmarks, driving_landmarks, frame_index, config
            )
        if config.flag_lip_retargeting:
            keypoints = self._apply_lip_retargeting(
                keypoints, source_landmarks, driving_landmarks, frame_index, config
            )
        return keypoints

    def _apply_eye_retargeting(
        self, keypoints, source_landmarks, driving_landmarks, frame_index: int, config
    ):
        eye_ratio = self.live_portrait_wrapper.calc_retargeting_ratio(
            source_landmarks, driving_landmarks
        )[0][frame_index]
        combined_eye_ratio = self.live_portrait_wrapper.calc_combined_eye_ratio(
            eye_ratio, source_landmarks
        )
        eye_delta = self.live_portrait_wrapper.retarget_eye(
            keypoints, combined_eye_ratio * config.eyes_retargeting_multiplier
        )
        return keypoints + eye_delta.reshape(-1, keypoints.shape[1], 3)

    def _apply_lip_retargeting(
        self, keypoints, source_landmarks, driving_landmarks, frame_index: int, config
    ):
        lip_ratio = self.live_portrait_wrapper.calc_retargeting_ratio(
            source_landmarks, driving_landmarks
        )[1][frame_index]
        combined_lip_ratio = self.live_portrait_wrapper.calc_combined_lip_ratio(
            lip_ratio, source_landmarks
        )
        lip_delta = self.live_portrait_wrapper.retarget_lip(
            keypoints, combined_lip_ratio * config.lip_retargeting_multiplier
        )
        return keypoints + lip_delta.reshape(-1, keypoints.shape[1], 3)

    def _warp_and_decode(self, source_features, keypoints):
        if self.live_portrait_wrapper.cfg.flag_stitching:
            keypoints = self.live_portrait_wrapper.stitching(
                source_features["transformed_kp"], keypoints
            )
        return self.live_portrait_wrapper.warp_decode(
            source_features["feature_3d"], source_features["transformed_kp"], keypoints
        )

    def _transform_and_blend(
        self,
        cropped_output: np.ndarray,
        source_frame: np.ndarray,
        crop_info: dict,
        config,
    ) -> np.ndarray:
        full_output = _transform_img(
            cropped_output,
            crop_info["M_c2o"],
            dsize=(source_frame.shape[1], source_frame.shape[0]),
        )

        if config.flag_pasteback:
            if config.mask_crop is None:
                config.mask_crop = cv2.imread(
                    make_abs_path("./utils/resources/mask_template.png"),
                    cv2.IMREAD_COLOR,
                )
            mask = (
                _transform_img(
                    config.mask_crop,
                    crop_info["M_c2o"],
                    dsize=(source_frame.shape[1], source_frame.shape[0]),
                ).astype(np.float32)
                / 255.0
            )
            return np.clip(
                mask * full_output + (1 - mask) * source_frame, 0, 255
            ).astype(np.uint8)

        return full_output
