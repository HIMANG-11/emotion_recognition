import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Ensure you are using the same TensorFlow version as training if possible
# Or a compatible version.

# Load the trained model
# Make sure the path to your model file is correct
try:
    model = load_model("emotion_recognition_cnn_rnn_model_final.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'emotion_recognition_cnn_rnn_model_final.h5' exists and is a valid Keras model file.")
    exit() # Exit if model loading fails

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Error loading Haar cascade classifier.")
    exit()

# Define the class names in the same order as trained by flow_from_directory
# flow_from_directory sorts classes alphabetically by folder name by default.
# Based on your training script, this order is likely correct:
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Real-time video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Camera opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert to grayscale as the model was trained on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = gray[y:y+h, x:x+w]

        # Resize face to match the model's input size (48x48)
        # Use INTER_AREA for shrinking, INTER_CUBIC or INTER_LINEAR for zooming
        if face.shape[0] < 48 or face.shape[1] < 48:
             face_resized = cv2.resize(face, (48, 48), interpolation=cv2.INTER_LINEAR)
        else:
             face_resized = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)


        # Normalize the face image - scale pixel values to [0, 1]
        face_normalized = face_resized / 255.0

        # Expand dimensions to match model input shape (batch_size, height, width, channels)
        # (48, 48) -> (1, 48, 48, 1)
        # The model's internal Reshape layer will handle the conversion to (1, 3, 768) for the LSTM
        face_expanded = np.expand_dims(face_normalized, axis=(0, -1))

        # Predict expression
        # The model expects a batch of images, face_expanded is a batch of 1 image
        predictions = model.predict(face_expanded, verbose=0) # verbose=0 to not print prediction progress per frame

        # Get the predicted class index and confidence
        label_index = np.argmax(predictions)
        confidence = np.max(predictions)
        label = class_names[label_index]

        # Display prediction and bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green rectangle
        # Display label and confidence
        # You can add a confidence threshold if you only want to show labels for high confidence predictions
        display_text = f'{label} ({confidence:.2f})'
        cv2.putText(frame, display_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2) # Blue text


    # Display the resulting frame
    cv2.imshow("Facial Expression Recognition", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()