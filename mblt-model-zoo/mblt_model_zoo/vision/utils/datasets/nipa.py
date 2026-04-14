# datasets/nipa.py

CLASSES = [
    "backhoe_loader", "cement_truck", "compactor", "dozer", "dump_truck",
    "excavator", "grader", "mobile_crane", "tower_crane", "wheel_loader",
    "worker", "Hardhat", "Red_Hardhat", "scaffolds", "Lifted Load",
    "Crane_Hook", "Hook"
]

DET_PALETTE = [
    (255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0), (0, 255, 0),
    (0, 255, 128), (0, 255, 255), (0, 128, 255), (0, 0, 255), (128, 0, 255),
    (255, 0, 255), (255, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 0),
    (0, 0, 128), (128, 128, 128)
]

def get_nipa_class_num() -> int:
    return len(CLASSES)

def get_nipa_label(idx: int) -> str:
    assert 0 <= idx < get_nipa_class_num(), f"Invalid index: {idx}"
    return CLASSES[idx]

def get_nipa_det_palette(idx: int) -> tuple:
    return DET_PALETTE[idx]
