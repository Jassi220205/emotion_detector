import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import numpy as np

# Load the pre-trained emotion detection model
get_custom_objects().update({'fbeta': lambda *args, **kwargs: None})
model = load_model('emotion_model.h5', compile=False)

# Emotion labels dictionary
emotion_dict = {
    0: "Angry 😠",
    1: "Disgust 🤢",
    2: "Fear 😨",
    3: "Happy 😊",
    4: "Sad 😢",
    5: "Surprise 😲",
    6: "Neutral 😐"
}

# Start webcam capture
cap = cv2.VideoCapture(0)

# Load Haar Cascade face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Read each frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        img = cropped_img.astype('float32') / 255.0
        img = np.reshape(img, (1, 48, 48, 1))

        # Predict emotion
        prediction = model.predict(img)
        emotion_index = int(np.argmax(prediction))
        emotion = emotion_dict[emotion_index]

        # Show what the model is seeing
        cv2.imshow("Cropped Face", cropped_img)

        # Print prediction confidence
        print("Prediction array:", prediction)
        print("Predicted emotion:", emotion)

        # Draw rectangle and emotion text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the video with emotion labels
    cv2.imshow("Real-Time Emotion Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
