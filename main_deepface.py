#Still requires some tinkering, currently non functional

import cv2
from deepface import DeepFace

# Initialize the USB camera capture
cap = cv2.VideoCapture(0)  # Use the appropriate camera index (e.g., 0 for the default camera)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open the camera")
    exit()

# Define the input size expected by the emotion model
input_width = 48
input_height = 48

# Load face detection model
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

    # Preprocess the frame for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = gray_frame[y:y+h, x:x+w]

        # Preprocess the face for emotion detection
        resized_face = cv2.resize(face, (input_width, input_height))

        # Save the resized face as an image file
        cv2.imwrite("temp_face.jpg", resized_face)

        # Perform face analysis using DeepFace
        try:
            result = DeepFace.analyze(img_path="temp_face.jpg", actions=['emotion'], detector_backend='opencv')
            emotion_label = result['dominant_emotion']

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display the emotion label on the frame
            cv2.putText(frame, f"Emotion: {emotion_label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except ValueError as e:
            print("Error:", str(e))
            continue

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
