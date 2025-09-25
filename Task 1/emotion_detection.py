import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

class EmotionDetector:
    def __init__(self, model_path='emotion_model.h5'):
        self.model_path = model_path
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = None
        self.load_model()

    def load_model(self):
        """Load pre-trained emotion detection model"""
        try:
            self.model = load_model(self.model_path)
            print("Emotion detection model loaded successfully")
        except:
            print(f"Could not load model from {self.model_path}")
            print("Please train the model first using the training notebook")

    def build_model(self):
        """Build CNN model for emotion detection"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

        model = Sequential([
            # First Conv Block
            Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # Second Conv Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # Third Conv Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # Fourth Conv Block
            Conv2D(512, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # Fully Connected Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')  # 7 emotions
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def preprocess_face(self, face_roi):
        """Preprocess face region for emotion detection"""
        # Resize to 48x48 and convert to grayscale
        face_roi = cv2.resize(face_roi, (48, 48))
        if len(face_roi.shape) == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Normalize pixel values
        face_roi = face_roi.astype('float32') / 255.0
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)

        return face_roi

    def detect_emotions(self, frame):
        """Detect emotions in faces within frame"""
        if self.model is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        emotions_detected = []

        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]

            # Preprocess for model
            processed_face = self.preprocess_face(face_roi)

            # Predict emotion
            emotion_prediction = self.model.predict(processed_face, verbose=0)
            emotion_probability = np.max(emotion_prediction)
            emotion_label = self.emotion_labels[np.argmax(emotion_prediction)]

            emotions_detected.append({
                'emotion': emotion_label,
                'confidence': emotion_probability,
                'location': (x, y, w, h)
            })

        return emotions_detected

    def draw_emotion_labels(self, frame, emotions):
        """Draw emotion labels on frame"""
        for emotion_data in emotions:
            x, y, w, h = emotion_data['location']
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Draw emotion label
            label = f"{emotion}: {confidence:.2f}"
            cv2.rectangle(frame, (x, y-30), (x+w, y), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def get_dominant_emotion(self, emotions):
        """Get the most confident emotion detection"""
        if not emotions:
            return "Neutral", 0.0

        best_emotion = max(emotions, key=lambda x: x['confidence'])
        return best_emotion['emotion'], best_emotion['confidence']
