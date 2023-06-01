import cv2

# Create a VideoCapture object for the camera
cap = cv2.VideoCapture(0)  # Use the appropriate device index (e.g., 0, 1, 2) if you have multiple cameras

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open the camera")
    exit()

# Start capturing and displaying frames
while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        break

    # Display the frame
    cv2.imshow("Video Stream", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()

# Destroy any OpenCV windows
cv2.destroyAllWindows()
