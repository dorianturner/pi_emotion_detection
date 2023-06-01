import cv2
import numpy as np
import tensorflow.lite as tflite

# Load the TFLite model
model_path = 'emotions_quarter_size.tflite'
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Load class labels
with open('labels.txt', 'r') as f:
    labels = f.read().splitlines()

# Initialize the USB camera capture
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_ANY)  # Use the appropriate camera index (e.g., 0 for the default camera)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open the camera")
    exit()

# Define the input size expected by the model
input_width = input_shape[2]
input_height = input_shape[1]

#Load face detection model
face_cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Read and process frames from the camera
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret will be True
    if not ret:
        print("Failed to receive frame from the camera")
        break

    # Preprocess the frame for input to the model
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = gray_frame[y:y+h, x:x+w]

        # Preprocess the face for input to the model
        resized_face = cv2.resize(face, (input_width, input_height))
        input_data = np.expand_dims(resized_face, axis=0)
        input_data = np.expand_dims(input_data, axis=3)
        input_data = input_data.astype(np.float32)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Retrieve the predicted class and confidence
        class_id = np.argmax(output_data)
        confidence = output_data[0][class_id]

        # Display the emotion label and confidence on the frame
        label = labels[class_id]
        cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
cv2.VideoCapture(0).release()
cv2.destroyAllWindows()
