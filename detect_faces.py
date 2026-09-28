from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo11n.pt")

input_folder = Path("images")
output_folder = Path("output")
output_folder.mkdir(exist_ok=True)

for image in input_folder.iterdir():
    if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        model.predict(
            source=str(image),
            save=True,
            project=str(output_folder),
            name="persons",
            exist_ok=True,
            classes=[0],
            conf=0.5
        )

print("Done! Person detection completed.")