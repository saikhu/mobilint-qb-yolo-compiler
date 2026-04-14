from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class ConTiLabYOLOv11_Set(ModelInfoSet):
    NIPA_V8 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    # "single": "/home/mobilint/Desktop/usman/train31/weights/yolov11_NIPA_Data_2025_v8.mxq",  # local path or update to TorchScript if exported
                    "global": "/workspace/weights/yolov11_NIPA_Data_2025_v8.mxq",  # Add your global mode model
                },
            },
        },
        pre_cfg={
            "Reader": {"style": "numpy"},
            "YoloPre": {"img_size": [1280, 1280]},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 14,  # your custom dataset's number of classes
            "nl": 3,   # adjust based on your model depth (ConTiLabYOLOv11 uses 3 detection layers)
            "names": [
                "cement_truck", "compactor", "dump_truck", "excavator", "grader", "mobile_crane", "tower_crane",  "Crane_Hook", "worker", "Hardhat", "Red_Hardhat", "scaffolds", "Lifted Load", "Hook"
            ],
        },
    )
    DEFAULT = NIPA_V8


class YOLO11s_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolo11s/aries/single/yolo11s.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolo11s/aries/multi/yolo11s.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolo11s/aries/global/yolo11s.mxq",
                },
                "regulus": {"single": None},
            },
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


class YOLO11m_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolo11m/aries/single/yolo11m.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolo11m/aries/multi/yolo11m.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolo11m/aries/global/yolo11m.mxq",
                },
                "regulus": {"single": None},
            },
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


class YOLO11l_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolo11l/aries/single/yolo11l.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolo11l/aries/multi/yolo11l.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolo11l/aries/global/yolo11l.mxq",
                },
                "regulus": {"single": None},
            },
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


class YOLO11x_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolo11x/aries/single/yolo11x.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolo11x/aries/multi/yolo11x.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolo11x/aries/global/yolo11x.mxq",
                },
                "regulus": {"single": None},
            },
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


def ConTiLabYOLOv11(
    local_path: str = None,  # or TorchScript .pt file
    model_type: str = "DEFAULT",
    infer_mode: str = "global",  # or "single"
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ConTiLabYOLOv11_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLO11s(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLO11s_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLO11m(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLO11m_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLO11l(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLO11l_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLO11x(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLO11x_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
