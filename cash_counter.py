import cv2
from ultralytics import YOLO
import supervision as sv

# Load image and model
image = cv2.imread('PATH_TO_IMAGE')
model = YOLO('PATH_TO_BEST_PT_FILE')
results = model.predict(image)

names = model.names
print(names)

# Denominations and their values
denominations = {
    'Penny': 0.01,
    'Nickel': 0.05,
    'Dime': 0.10,
    'Quarter': 0.25,
    'one': 1.00,
    'five': 5.00,
    'ten': 10.00,
    'twenty': 20.00,
    'fifty': 50.00,
    'hundred': 100.00
}

# Initialize a dictionary to count each denomination
denomination_counts = {key: 0 for key in denominations.keys()}
print(denomination_counts)
# Count each denomination detected
for r in results:
    for c in r.boxes.cls:
        print(c)
        class_name = model.names[int(c)]
        if class_name in denomination_counts:
            denomination_counts[class_name] += 1

# Calculate the total value
total_value = sum(denominations[name] * count for name, count in denomination_counts.items())

print(f"Total Dollar Amount: ${total_value:.2f}")

detections = sv.Detections.from_ultralytics(results)

bounding_box_annotator = sv.BoundingBoxAnnotator()
    annotated_frame = bounding_box_annotator.annotate(
    scene=image.copy(),
    detections=detections)
cv2.putText(annotated_frame, f"You have ${total_value}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
sv.plot(annotated_frame, (12,12))

# model.predict(source=0, show = True, conf = 0.8)
