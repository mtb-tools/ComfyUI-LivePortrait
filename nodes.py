from pathlib import Path
import torch
import yaml
import folder_paths
import comfy.model_management as mm
import comfy.utils
from .liveportrait.live_portrait_pipeline import LivePortraitPipeline
from .liveportrait.config.crop_config import CropConfig
from .liveportrait.utils.cropper import Cropper
from .liveportrait.modules.spade_generator import SPADEDecoder
from .liveportrait.modules.warping_network import WarpingNetwork
from .liveportrait.modules.motion_extractor import MotionExtractor
from .liveportrait.modules.appearance_feature_extractor import (
    AppearanceFeatureExtractor,
)
from .liveportrait.modules.stitching_retargeting_network import (
    StitchingRetargetingNetwork,
)
from .log import logger

script_directory = Path(__file__).parent
"""Path to the extension's root."""

lp_root = script_directory / "liveportrait"
"""Upsream source code root"""

models_dir = Path(folder_paths.models_dir) / "liveportrait"
"""LivePortrait model directory."""


def apply_config(base, **kwargs):
    for k, v in kwargs.items():
        setattr(base, k, v)


class InferenceConfig:
    def __init__(
        self,
        mask_crop=None,
        flag_use_half_precision=True,
        flag_lip_zero=True,
        lip_zero_threshold=0.03,
        flag_eye_retargeting=False,
        flag_lip_retargeting=False,
        flag_stitching=True,
        flag_relative=True,
        anchor_frame=0,
        input_shape=(256, 256),
        flag_write_result=True,
        flag_pasteback=True,
        ref_max_shape=1280,
        ref_shape_n=2,
        device_id=0,
        flag_do_crop=True,
        flag_do_rot=True,
    ):
        self.flag_use_half_precision = flag_use_half_precision
        self.flag_lip_zero = flag_lip_zero
        self.lip_zero_threshold = lip_zero_threshold
        self.flag_eye_retargeting = flag_eye_retargeting
        self.flag_lip_retargeting = flag_lip_retargeting
        self.flag_stitching = flag_stitching
        self.flag_relative = flag_relative
        self.anchor_frame = anchor_frame
        self.input_shape = input_shape
        self.flag_write_result = flag_write_result
        self.flag_pasteback = flag_pasteback
        self.ref_max_shape = ref_max_shape
        self.ref_shape_n = ref_shape_n
        self.device_id = device_id
        self.flag_do_crop = flag_do_crop
        self.flag_do_rot = flag_do_rot
        self.mask_crop = mask_crop


class DownloadAndLoadLivePortraitModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "precision": (
                    [
                        "fp16",
                        "fp32",
                    ],
                    {"default": "fp16"},
                ),
            },
        }

    RETURN_TYPES = ("LIVEPORTRAITPIPE",)
    RETURN_NAMES = ("live_portrait_pipe",)
    FUNCTION = "load_model"
    CATEGORY = "LivePortrait"

    def load_model(self, precision="fp16"):
        device = mm.get_torch_device()
        mm.soft_empty_cache()

        pbar = comfy.utils.ProgressBar(5)

        # NOTE: Download model
        if not models_dir.exists():
            logger.info(f"Downloading model to: {models_dir}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="Kijai/LivePortrait_safetensors",
                local_dir=models_dir.as_posix(),
                local_dir_use_symlinks=False,
            )

        model_config_path = lp_root / "config" / "models.yaml"

        with open(model_config_path, "r") as file:
            model_config = yaml.safe_load(file)

        models = {}
        model_specs = [
            (
                "appearance_feature_extractor",
                AppearanceFeatureExtractor,
                "appearance_feature_extractor_params",
            ),
            ("motion_extractor", MotionExtractor, "motion_extractor_params"),
            ("warping_module", WarpingNetwork, "warping_module_params"),
            ("spade_generator", SPADEDecoder, "spade_generator_params"),
        ]
        for name, model_class, params_key in model_specs:
            model_params = model_config["model_params"][params_key]
            model = model_class(**model_params).to(device)
            model.load_state_dict(
                comfy.utils.load_torch_file(
                    (models_dir / f"{name}.safetensors").as_posix()
                )
            )
            model.eval()
            models[name] = model
            logger.info(f"Load {name} done.")
            pbar.update(1)

        # NOTE: Stitch module
        def filter_checkpoint_for_model(checkpoint, prefix):
            return {
                key.replace(f"{prefix}_module.", ""): value
                for key, value in checkpoint.items()
                if key.startswith(prefix)
            }

        stitch_cfg = model_config["model_params"]["stitching_retargeting_module_params"]
        checkpoint = comfy.utils.load_torch_file(
            (models_dir / "stitching_retargeting_module.safetensors").as_posix()
        )

        stitching_retargeting_module = {}
        for prefix, config_key in [
            ("retarget_shoulder", "stitching"),
            ("retarget_mouth", "lip"),
            ("retarget_eye", "eye"),
        ]:
            filtered_checkpoint = filter_checkpoint_for_model(checkpoint, prefix)
            model = StitchingRetargetingNetwork(**stitch_cfg.get(config_key))
            model.load_state_dict(filtered_checkpoint)
            model = model.to(device)
            model.eval()
            stitching_retargeting_module[config_key] = model

        logger.info("Load stitching_retargeting_module done.")
        pbar.update(1)

        pipeline = LivePortraitPipeline(
            **models,
            stitching_retargeting_module=stitching_retargeting_module,
            inference_cfg=InferenceConfig(
                device_id=device,
                flag_use_half_precision=True if precision == "fp16" else False,
            ),
        )

        pbar.update(1)

        return (pipeline,)


# OUR CURRENT NODE
class LivePortraitProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("LIVEPORTRAITPIPE",),
                "source_image": ("IMAGE",),
                "driving_images": ("IMAGE",),
                "dsize": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "scale": (
                    "FLOAT",
                    {"default": 1.6, "min": 1.0, "max": 4.0, "step": 0.01},
                ),
                "vx_ratio": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "vy_ratio": (
                    "FLOAT",
                    {"default": -0.125, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "lip_zero": ("BOOLEAN", {"default": True}),
                "eye_retargeting": ("BOOLEAN", {"default": False}),
                "eyes_retargeting_multiplier": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.001},
                ),
                "lip_retargeting": ("BOOLEAN", {"default": False}),
                "lip_zero_threshold": (
                    "FLOAT",
                    {"default": 0.03, "min": 0, "step": 0.001},
                ),
                "lip_retargeting_multiplier": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.001},
                ),
                "anchor_frame": ("INT", {"default": 0}),
                "do_rot": ("BOOLEAN", {"default": True}),
                "do_crop": ("BOOLEAN", {"default": True}),
                "stitching": ("BOOLEAN", {"default": True}),
                "relative": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mismatch_method": (
                    ["repeat", "cycle", "mirror", "nearest"],
                    {"default": "repeat"},
                ),
                "onnx_device": (
                    [
                        "CPU",
                        "CUDA",
                    ],
                    {"default": "CPU"},
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "cropped_images",
        "full_images",
    )
    FUNCTION = "process"
    CATEGORY = "LivePortrait"

    def process(
        self,
        source_image: torch.Tensor,
        driving_images: torch.Tensor,
        dsize: int,
        scale: float,
        vx_ratio: float,
        vy_ratio: float,
        pipeline: LivePortraitPipeline,
        lip_zero: bool,
        eye_retargeting: bool,
        lip_retargeting: bool,
        do_crop: bool,
        do_rot: bool,
        stitching: bool,
        relative: bool,
        eyes_retargeting_multiplier: float,
        lip_retargeting_multiplier: float,
        lip_zero_threshold: float,
        anchor_frame: int = 0,
        mismatch_method: str = "repeat",
        onnx_device="CUDA",
    ):
        source_np = (source_image * 255).byte().numpy()
        driving_images_np = (driving_images * 255).byte().numpy()

        crop_cfg = CropConfig(
            dsize=dsize,
            scale=scale,
            vx_ratio=vx_ratio,
            vy_ratio=vy_ratio,
        )

        cropper = Cropper(crop_cfg=crop_cfg, provider=onnx_device)
        pipeline.cropper = cropper

        apply_config(
            pipeline.live_portrait_wrapper.cfg,
            flag_eye_retargeting=eye_retargeting,
            eyes_retargeting_multiplier=eyes_retargeting_multiplier,
            flag_lip_retargeting=lip_retargeting,
            lip_retargeting_multiplier=lip_retargeting_multiplier,
            flag_stitching=stitching,
            flag_relative=relative,
            flag_lip_zero=lip_zero,
            lip_zero_threshold=lip_zero_threshold,
            anchor_frame=anchor_frame,
            flag_do_crop=do_crop,
            flag_do_rot=do_rot,
        )

        cropped_out_list = []
        full_out_list = []

        cropped_out_list, full_out_list = pipeline.execute(
            source_np, driving_images_np, mismatch_method
        )
        cropped_tensors_out = (
            torch.stack([torch.from_numpy(np_array) for np_array in cropped_out_list])
            / 255
        )
        full_tensors_out = (
            torch.stack([torch.from_numpy(np_array) for np_array in full_out_list])
            / 255
        )

        return (cropped_tensors_out.cpu().float(), full_tensors_out.cpu().float())


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadLivePortraitModels": DownloadAndLoadLivePortraitModels,
    "LivePortraitProcess": LivePortraitProcess,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadLivePortraitModels": "(Down)Load LivePortraitModels",
}
