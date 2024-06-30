import cv2
import numpy as np
import datetime
import matplotlib.pyplot as plt
import gradio as gr

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_objects(image):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    return [(boxes[i], class_ids[i], confidences[i]) for i in range(len(boxes)) if i in indexes]

def capture_frame(frame):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cv2.imwrite(f"captured_frame_{current_time}.jpg", frame)
    print(f"Captured frame saved as 'captured_frame_{current_time}.jpg'")

def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detections = detect_objects(image)
    for (box, class_id, confidence) in detections:
        x, y, w, h = box
        label = str(classes[class_id])
        color = (0, 255, 0) if label == "person" else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def capture_and_process():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_image(frame)
        yield processed_frame
    cap.release()

# Define Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# YOLO Object Detection")
    gr.Markdown("## Real-time object detection using YOLO")

    with gr.Tab("Live Camera Feed"):
        gr.Markdown("Press the button to start the live camera feed with real-time object detection.")
        live_output = gr.Image(type="numpy", label="Live Camera Feed")
        gr.Button("Start Live Camera").click(capture_and_process, outputs=live_output)

    with gr.Tab("Upload Image"):
        gr.Markdown("Upload an image and the YOLO model will detect objects in the image, highlighting humans.")
        image_input = gr.Image(type="numpy", label="Upload an image")
        image_output = gr.Image(type="numpy", label="Detected objects")
        image_input.upload(process_image, inputs=image_input, outputs=image_output)

# Launch Gradio interface
iface.launch()
