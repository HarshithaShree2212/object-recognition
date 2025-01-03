import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load a pre-trained YOLO model from Torch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(image_rgb)

    # Parse results
    detections = results.pandas().xyxy[0]  # Bounding boxes with class and confidence

    # Draw bounding boxes and count objects
    for _, row in detections.iterrows():
        x1, y1, x2, y2, confidence, cls, label = row
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Count objects
    object_counts = detections['name'].value_counts().to_dict()

    # Display results
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    print("Object Counts:", object_counts)

# Provide the path to your image
image_path = "path_to_your_image.jpg"
detect_objects(image_path)
