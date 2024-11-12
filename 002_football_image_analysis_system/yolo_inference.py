from ultralytics import YOLO

# model = YOLO('yolo11x')
model = YOLO("training/runs/detect/train/weights/best.pt")

results = model.predict("./input_videos/08fd33_4.mp4", save=True)

print(results[0])
print(40 * "=")

for box in results[0].boxes:
    print(box)
