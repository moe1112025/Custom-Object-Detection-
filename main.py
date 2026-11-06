
from pathlib import Path
from ultralytics import YOLO


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent

# Use a pretrained YOLOv8 model
MODEL_PATH = "yolov8n.pt"

# To use an existing custom YOLO model, replace the line above:
# MODEL_PATH = ROOT_DIR / "best.pt"

INPUT_IMAGE = ROOT_DIR / "sample.png"

OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_IMAGE = OUTPUT_DIR / "prediction.png"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640


# ==========================================
# 2. OBJECT DETECTION
# ==========================================

def main():

    print("====================================")
    print("      YOLOv8 IMAGE DETECTION")
    print("====================================")

    # Check whether the input image exists
    if not INPUT_IMAGE.is_file():
        raise FileNotFoundError(
            f"Input image not found: {INPUT_IMAGE}"
        )

    # Check whether a custom model file exists
    if isinstance(MODEL_PATH, Path) and not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Custom model not found: {MODEL_PATH}"
        )

    # Create the output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the YOLO model
    print(f"\n[INFO] Loading model: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    # Run object detection on the input image
    print(f"[INFO] Analyzing image: {INPUT_IMAGE.name}")

    results = model.predict(
        source=str(INPUT_IMAGE),
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    # Get the first image result
    result = results[0]

    # Save the annotated image
    result.save(filename=str(OUTPUT_IMAGE))

    # Read detected bounding boxes
    boxes = result.boxes

    print("\n--- Detection Summary ---")

    if boxes is None or len(boxes) == 0:

        print("No objects detected.")

    else:

        print(f"Total objects detected: {len(boxes)}")

        for index, box in enumerate(boxes, start=1):

            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]

            confidence = float(box.conf[0].item()) * 100

            x1, y1, x2, y2 = (
                int(value) for value in box.xyxy[0].tolist()
            )

            print(f"\nObject {index}")
            print(f"Class: {class_name}")
            print(f"Confidence: {confidence:.2f}%")
            print(f"Bounding Box: ({x1}, {y1}, {x2}, {y2})")

    print("\n====================================")
    print("[SUCCESS] Detection completed.")
    print(f"[OUTPUT] {OUTPUT_IMAGE}")
    print("====================================")


# ==========================================
# 3. SCRIPT ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
