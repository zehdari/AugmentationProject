import supervision as sv
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

# Load the model
model = RFDETRNano()

# Load the image
image_file = "images/dog.jpg"
image = Image.open(image_file)

# Perform inference
detections = model.predict(image, threshold=0.5)

# Annotate the image
labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

# Display the result
sv.plot_image(annotated_image)
