from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class CustomYOLOv8_Set(ModelInfoSet):
    NIPA_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "/home/mobilint/Desktop/mobilint_aries/repos/mblt-model-zoo/yolov8_sib/yolov8l.mxq",  # local path or update to TorchScript if exported
                    "global": "/home/mobilint/Desktop/mobilint_aries/repos/mblt-model-zoo/yolov8_sib/best_global_yolov8l.mxq",  # Add your global mode model
                },
            },
        },
        pre_cfg={
            "Reader": {"style": "numpy"},
            "YoloPre": {"img_size": [640, 640]},
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 17,  # your custom dataset's number of classes
            "nl": 3,   # adjust based on your model depth (YOLOv8l uses 3 detection layers)
            "names": [
                "backhoe_loader", "cement_truck", "compactor", "dozer", "dump_truck", 
                "excavator", "grader", "mobile_crane", "tower_crane", "wheel_loader", 
                "worker", "Hardhat", "Red_Hardhat", "scaffolds", "Lifted Load", 
                "Crane_Hook", "Hook"
            ],
        },
    )
    DEFAULT = NIPA_V1


class YOLOv8s_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov8s/aries/single/yolov8s.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov8s/aries/multi/yolov8s.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov8s/aries/global/yolov8s.mxq",
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


class YOLOv8m_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov8m/aries/single/yolov8m.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov8m/aries/multi/yolov8m.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov8m/aries/global/yolov8m.mxq",
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


class YOLOv8l_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov8l/aries/single/yolov8l.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov8l/aries/multi/yolov8l.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov8l/aries/global/yolov8l.mxq",
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


class YOLOv8x_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/object_detection/yolov8x/aries/single/yolov8x.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/object_detection/yolov8x/aries/multi/yolov8x.mxq",
                    "global": "https://dl.mobilint.com/model/vision/object_detection/yolov8x/aries/global/yolov8x.mxq",
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



def CustomYOLOv8(
    local_path: str = None,  # or TorchScript .pt file
    model_type: str = "DEFAULT",
    infer_mode: str = "global",  # or "single"
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        CustomYOLOv8_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )



def YOLOv8s(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv8s_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv8m(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv8m_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv8l(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv8l_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def YOLOv8x(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        YOLOv8x_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
