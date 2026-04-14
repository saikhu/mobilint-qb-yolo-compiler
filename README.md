# Mobilint-qb YOLO compiler

End-to-end workflow to **convert, calibrate, and compile** a custom **YOLOv11** model to **`.mxq`** using **Mobilint `qb_compiler` / `qubee`** for the **ARIES NPU**, plus simple image/video inference utilities.

This repo captures the exact steps I used to compile:

* **Model**: `best.pt` (Ultralytics YOLO)
* **Input size**: `1280 x 1280`
* **Classes (nc=14)**:

  * `cement_truck, compactor, dump_truck, excavator, grader, mobile_crane, tower_crane, Crane_Hook, worker, Hardhat, Red_Hardhat, scaffolds, Lifted Load, Hook`

---

## What’s inside

* `00_convert_model.py` – export `best.pt → ONNX`
* `random_copy_images.py` – sample calibration images
* `01_prepare_calib_dataset.py` – preprocess + build calibration dataset
* `02_compile_model.py` / `02_compile_model_global.py` – compile to `.mxq`
* `03_test.py`, `04_batch_inference.py`, `04_video_inference.py`, `05_infer_video_tree_mblt.py` – validation utilities
* `mblt-model-zoo/` – local copy of Mobilint model zoo with a **custom ConTiLabYOLOv11 entry**
* `weights/` – model artifacts (`best.pt`, `.onnx`, `.mxq`)

---

## Requirements

### For compilation

* Linux host with:

  * **Docker**
  * **NVIDIA GPU + nvidia-container-toolkit**
* Mobilint compiler image:

  * `mobilint/qbcompiler:0.10-...` (tag may vary by CUDA/Ubuntu build)
  > For the YOLOv11 model I used `mobilint/qbcompiler:0.10-cuda12.8.1-ubuntu22.04`

```bash
docker run -it --gpus all --ipc=host --name qb_compiler \
  -v "$(pwd)":/workspace \
  mobilint/qbcompiler:0.10-cuda12.8.1-ubuntu22.04 \
  /bin/bash
```


### For inference

* A PC with:

  * **ARIES NPU installed**
  * **MACCEL SW installed**
* You only need Docker for compilation; inference can run on the host.

---

## Repo structure (simplified)

```text
.
├── 00_convert_model.py
├── random_copy_images.py
├── 01_prepare_calib_dataset.py
├── 02_compile_model.py
├── 02_compile_model_global.py
├── 03_test.py
├── 04_batch_inference.py
├── 04_video_inference.py
├── 05_infer_video_tree_mblt.py
├── args.yaml
├── qubee-0.10.0.0+aries2.cuda12.8-py3-none-any.whl
├── mblt-model-zoo/
└── weights/
    ├── best.pt
    ├── best_NIPA_Data_2025_v8_train31.onnx
    └── yolov11_NIPA_Data_2025_v8.mxq
```

---

## Quickstart

### 1) Start the compiler container
```bash
git clone https://github.com/contilabai/mobilint-qb-yolo-compiler.git
cd mobilint-qb-yolo-compiler
```

```bash
docker run -it --gpus all --ipc=host --name qb_compiler \
  -v "$(pwd)":/workspace \
  mobilint/qbcompiler:0.10-cuda12.8.1-ubuntu22.04 \
  /bin/bash
```

Inside the container (if needed):

```bash
cd /workspace
pip install ./qubee-0.10.0.0+aries2.cuda12.8-py3-none-any.whl
```

---

## Workflow

### Step 0 — Place your trained weights

```text
weights/best.pt
```

---

### Step 1 — Export ONNX

`00_convert_model.py`:

```python
from ultralytics import YOLO
import os

model = YOLO("./weights/best.pt")
model.export(format="onnx")

os.rename("./weights/best.onnx", "./weights/best_NIPA_Data_2025_v8_train31.onnx")
```

Run:

```bash
python 00_convert_model.py
```

Output:

```text
weights/best_NIPA_Data_2025_v8_train31.onnx
```

---

### Step 2 — Sample calibration images

`random_copy_images.py` (edit `src_dir`, `num_to_copy` as needed):

```bash
python random_copy_images.py
```

Output:

```text
calibration_images/
```

---

### Step 3 — Build calibration dataset

`01_prepare_calib_dataset.py` uses a YOLO-style letterbox preprocess to `1280x1280` and saves float32 normalized inputs.

Run:

```bash
python 01_prepare_calib_dataset.py
```

Output:

```text
calibrated_dataset/NIPA_Data_2025_v8
```

---

### Step 4 — Compile to `.mxq`

`02_compile_model.py` (core settings you used):

```python
from qubee import mxq_compile

onnx_model_path = "weights/best_NIPA_Data_2025_v8_train31.onnx"
calib_data_path = "calibrated_dataset/NIPA_Data_2025_v8"

mxq_compile(
    model=onnx_model_path,
    calib_data_path=calib_data_path,
    quantize_method="maxpercentile",
    is_quant_ch=True,
    quantize_percentile=0.999,
    topk_ratio=0.01,
    quant_output="ch",
    save_path="yolov11_NIPA_Data_2025_v8.mxq",
    backend="onnx",
    target_device="aries",
    inference_scheme="global"
)
```

Run:

```bash
python 02_compile_model.py
```

Output:

```text
weights/yolov11_NIPA_Data_2025_v8.mxq
```
---

## Using the local `mblt-model-zoo` wrapper

`mblt_model_zoo/vision/object_detection/yolo11.py`

### Factory update

```python
def ConTiLabYOLOv11(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        ConTiLabYOLOv11_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
```

### Custom ModelInfo entry

```python
class ConTiLabYOLOv11_Set(ModelInfoSet):
    NIPA_V8 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "global": "/workspace/weights/yolov11_NIPA_Data_2025_v8.mxq",
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
            "nc": 14,
            "nl": 3,
            "names": [
                "cement_truck", "compactor", "dump_truck", "excavator",
                "grader", "mobile_crane", "tower_crane", "Crane_Hook",
                "worker", "Hardhat", "Red_Hardhat", "scaffolds",
                "Lifted Load", "Hook"
            ],
        },
    )
    DEFAULT = NIPA_V8
```

```python
from mblt_model_zoo.vision import ConTiLabYOLOv11

yolo = ConTiLabYOLOv11(
    local_path="weights/yolov11_NIPA_Data_2025_v8.mxq",
    model_type="DEFAULT",
    infer_mode="global"
)
```

---

## Video inference

Use whichever script matches your setup:

```bash
python 04_video_inference.py --model_path weights/yolov11_NIPA_Data_2025_v8.mxq
python 05_infer_video_tree_mblt.py
```

---

## Adapting this repo for a new dataset

Update these places:

1. **Class names / count**

* Update `nc` and `names` in your custom `ModelInfo`.

2. **Input size**

* Keep consistent across:

  * `01_prepare_calib_dataset.py` (`img_size`)
  * `YoloPre` config in model zoo
  * Your training/export settings (if fixed-shape)

3. **File naming**

* Update:

  * ONNX rename target
  * calibration `save_name`
  * `.mxq` `save_path`

---

## Common pitfalls

* **Mismatch of `img_size`** between calibration preprocess and inference preprocess can cause poor accuracy.
* Ensure you're using:

  * `target_device="aries"`
  * `inference_scheme="global"` if you compiled for global mode.
* Verify the calibration path exists:

  ```bash
  ls calibrated_dataset/NIPA_Data_2025_v8
  ```

---

## Notes

* This repo assumes an **Ultralytics YOLOv11**-style export and a **custom NIPA V8 dataset**.
* The included `mblt-model-zoo` folder is treated as a **local dependency** for quick iteration and custom model registration.

---
