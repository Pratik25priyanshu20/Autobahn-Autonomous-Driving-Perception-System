import cv2

from src.perception.segmentation.deeplabv3_segmenter import DeepLabV3Segmenter
from src.perception.segmentation.postprocess import extract_drivable_area


def main():
    video_path = "data/samples/test_drive.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to read first frame from video")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    segmenter = DeepLabV3Segmenter(device="cpu")
    out = segmenter.infer(frame)
    drivable = extract_drivable_area(out["mask"])

    print("✅ Segmentation OK")
    print("Mask shape:", out["mask"].shape)
    print("Confidence:", round(out["confidence"], 4))
    print("Latency (ms):", round(out["latency_ms"], 2))
    print("Drivable area pixels:", int(drivable.sum()))


if __name__ == "__main__":
    main()
