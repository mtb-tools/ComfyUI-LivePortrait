import folder_paths
from pathlib import Path
from .path import LivePortraitPipeline, Cropper
from .liveportrait.src.config.crop_config import CropConfig
import torch
import pickle
import numpy as np
import comfy.utils


class LoadMotionTemplate:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = Path(folder_paths.get_input_directory()) / "liveportrait"
        files = [x.stem for x in input_dir.glob("*.pkl")]
        return {
            "required": {
                "template": (sorted(files),),
            }
        }

    RETURN_TYPES = ("LIVEPORTRAIT_TEMPLATE",)
    RETURN_NAMES = ("template",)
    FUNCTION = "process"
    CATEGORY = "LivePortrait"

    def process(self, template: str):
        input_dir = Path(folder_paths.get_input_directory()) / "liveportrait"

        return ((input_dir / f"{template}.pkl").as_posix(),)


class LivePortraitMotionTemplate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("LIVEPORTRAITPIPE",),
                "name": ("STRING", {"default": "MotionTemplate"}),
                "driving_images": ("IMAGE",),
                "dsize": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "scale": ("FLOAT", {"default": 2.3, "min": 1.0, "max": 4.0}),
                "vx_ratio": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "vy_ratio": (
                    "FLOAT",
                    {"default": -0.125, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("LIVEPORTRAIT_TEMPLATE",)
    RETURN_NAMES = ("template",)
    FUNCTION = "process"
    CATEGORY = "LivePortrait"

    def process(
        self,
        name: str,
        pipeline: LivePortraitPipeline,
        driving_images: torch.Tensor,
        dsize: int,
        scale: float,
        vx_ratio: float,
        vy_ratio: float,
    ):
        driving_images_np = (driving_images * 255).byte().numpy()

        crop_cfg = CropConfig(
            dsize=dsize, scale=scale, vx_ratio=vx_ratio, vy_ratio=vy_ratio
        )
        # inference_cfg = InferenceConfig()
        # )  # use attribute of args to initial InferenceConfig

        # wrapper = LivePortraitWrapper(cfg=inference_cfg)
        cropper = Cropper(crop_cfg=crop_cfg)

        # wants BGR

        resized = [cv2.resize(im, (256, 256)) for im in driving_images_np]
        lmk = cropper.get_retargeting_lmk_info(resized)
        prepared = pipeline.live_portrait_wrapper.prepare_driving_videos(resized)

        count = prepared.shape[0]

        templates = []

        progress = comfy.utils.ProgressBar(count)

        for i in range(count):
            id = prepared[i]
            kp_info = pipeline.live_portrait_wrapper.get_kp_info(id)
            rot = get_rotation_matrix(kp_info["pitch"], kp_info["yaw"], kp_info["roll"])

            template_dct = {"n_frames": count, "frames_index": i}
            template_dct["scale"] = kp_info["scale"].cpu().numpy().astype(np.float32)
            template_dct["R_d"] = rot.cpu().numpy().astype(np.float32)
            template_dct["exp"] = kp_info["exp"].cpu().numpy().astype(np.float32)
            template_dct["t"] = kp_info["t"].cpu().numpy().astype(np.float32)
            progress.update(1)

            templates.append(template_dct)

        out_dir = Path(folder_paths.get_input_directory()) / "liveportrait"
        out_dir.mkdir(exist_ok=True, parents=True)
        res = out_dir / (name + ".pkl")
        with open(res, "wb") as f:
            pickle.dump([templates, lmk], f)

        return (res.as_poxix(),)
