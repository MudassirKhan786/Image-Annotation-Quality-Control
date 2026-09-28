from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo11n.pt")

input_folder = Path("images")

for image in input_folder.iterdir():
    if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        results = model.predict(
            source=str(image),
            classes=[0],
            conf=0.25
        )

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])

                if confidence < 0.50:
                    print(f"REVIEW: {image.name} - Confidence: {confidence:.2f}")
                else:
                    print(f"OK: {image.name} - Confidence: {confidence:.2f}")

print("Quality checking completed.")