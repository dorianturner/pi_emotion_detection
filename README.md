#  PI Emotion Detection

This project aims for real time emotion detection from a camera to be used as a way to activate devices connected to the pi via gpio pins.

## Usage
clone repository with: ```git clone https://github.com/dorianturner/pi_emotion_detection.git```  

run main_tflite with python3: ```python3 main_tflite.py```

## Notes
This is using **64-bit** raspberry pi os and a usb webcam with the video handling and face detection coming from opencv. 

## TODO
Get the deepface script working and train a bettermodel and get the h5_to_tflite model converter working.
     
