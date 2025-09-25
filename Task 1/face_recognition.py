import cv2
import numpy as np
import face_recognition
import pickle
import os
from datetime import datetime

class FaceRecognizer:
    def __init__(self, encodings_path='face_encodings.pkl'):
        self.known_encodings = []
        self.known_names = []
        self.encodings_path = encodings_path
        self.load_encodings()

    def load_encodings(self):
        """Load pre-trained face encodings"""
        if os.path.exists(self.encodings_path):
            with open(self.encodings_path, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data['encodings']
                self.known_names = data['names']
            print(f"Loaded {len(self.known_names)} face encodings")
        else:
            print("No encodings file found. Please train the model first.")

    def save_encodings(self):
        """Save face encodings to file"""
        data = {
            'encodings': self.known_encodings,
            'names': self.known_names
        }
        with open(self.encodings_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved encodings for {len(self.known_names)} faces")

    def train_from_folder(self, dataset_path):
        """Train face recognition model from folder structure:
        dataset/
        ├── person1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── person2/
            ├── image1.jpg
            └── image2.jpg
        """
        self.known_encodings = []
        self.known_names = []

        for person_name in os.listdir(dataset_path):
            person_folder = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_folder):
                continue

            print(f"Processing {person_name}...")

            for image_file in os.listdir(person_folder):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_folder, image_file)

                    # Load and encode face
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)

                    if face_encodings:
                        self.known_encodings.append(face_encodings[0])
                        self.known_names.append(person_name)

        self.save_encodings()
        print(f"Training completed! Encoded {len(self.known_names)} faces")

    def recognize_faces(self, frame):
        """Recognize faces in a frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        recognized_faces = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)

            name = "Unknown"
            confidence = 0

            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]

            recognized_faces.append({
                'name': name,
                'confidence': confidence,
                'location': face_location
            })

        return recognized_faces

    def draw_face_boxes(self, frame, recognized_faces):
        """Draw bounding boxes and names on frame"""
        for face in recognized_faces:
            top, right, bottom, left = face['location']
            name = face['name']
            confidence = face['confidence']

            # Draw rectangle
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw name and confidence
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else name
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        return frame
