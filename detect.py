import cv2
import numpy as np
#import serial

# Load the YOLOv4 Tiny model
net = cv2.dnn.readNet("backup/best2.weights", "cfg/best2.cfg")

# Load the class labels
with open("data/obj1.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

#ser = serial.Serial("/dev/ttyACM0", 9600)


# Get the output layer names of the YOLOv4 model
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Set video dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def darknet_helper(frame, width, height):
    # Prepare input blob and perform forward pass through the YOLOv4 model
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (width, height), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    # Process the outputs and extract bounding box information
    detections = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                detections.append((classes[class_id], confidence, (x, y, x+w, y+h)))

    # Calculate width and height ratios for resizing bounding boxes
    width_ratio = width / 640
    height_ratio = height / 480

    return detections, width_ratio, height_ratio

detected_object = False

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    #frame = cv2.flip(frame, 0)
    if not ret:
        break

    # Process frame and draw bounding boxes
    detections, width_ratio, height_ratio = darknet_helper(frame, 320, 320)

    # Check if any detections were found
    # if len(detections) == 0:
    #     label = "green"
    #     confidence = 0.0
    #     bbox = None
    #     detections.append((label, confidence, bbox))
    
    # Iterate over the detections and draw bounding boxes on the frame
    for label, confidence, bbox in detections:
        print(f"Label: {label}")

        if 'green' in label:
            label = 'g'
        elif 'red' in label:
            label = 'r'
        elif 'yellow' in label:
            label = 'y'
        
        #ser.write(label.encode('utf-8'))

        x, y, x_plus_w, y_plus_h = bbox
        x = int(x * width_ratio)
        y = int(y * height_ratio)
        x_plus_w = int(x_plus_w * width_ratio)
        y_plus_h = int(y_plus_h * height_ratio)

        # Draw the bounding box rectangle and label on the frame
        cv2.rectangle(frame, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with bounding boxes
    cv2.imshow("Object Detection", frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) == ord('q'):
        break



#ser.close()
    

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
