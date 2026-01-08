import cv2
import numpy as np

# Cityscapes class IDs (torchvision mapping)
ROAD = 0
SIDEWALK = 1
LANE_MARKING = 2  # approximate, depending on model

DRIVABLE_CLASSES = {ROAD}


def extract_drivable_area(mask: np.ndarray) -> np.ndarray:
    """
    Convert semantic segmentation mask into binary drivable area.

    Args:
        mask: (H, W) int class ids

    Returns:
        drivable: (H, W) uint8 mask {0,1}
    """
    drivable = np.zeros_like(mask, dtype=np.uint8)

    for cls in DRIVABLE_CLASSES:
        drivable[mask == cls] = 1

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    drivable = cv2.morphologyEx(drivable, cv2.MORPH_CLOSE, kernel)
    drivable = cv2.morphologyEx(drivable, cv2.MORPH_OPEN, kernel)

    return drivable
