import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
import face_recognition
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import json
import pickle
from datetime import datetime
import imutils

class DrowsinessDetector:
    def __init__(self, age_model_path='models/age_model.h5'):
        # Initialize face detection methods
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize dlib face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self.load_shape_predictor()

        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        # Eye aspect ratio threshold for drowsiness
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 20

        # Age estimation model
        self.age_model = None
        self.age_model_path = age_model_path
        self.load_age_model()

        # Counters
        self.drowsy_counter = 0
        self.total_people_detected = 0
        self.sleeping_people_count = 0

    def load_shape_predictor(self):
        """Load dlib shape predictor for facial landmarks"""
        predictor_path = 'models/shape_predictor_68_face_landmarks.dat'

        if os.path.exists(predictor_path):
            self.predictor = dlib.shape_predictor(predictor_path)
            print("✅ Dlib shape predictor loaded")
        else:
            print("⚠️ Dlib shape predictor not found. Download from:")
            print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("Extract and place in models/ directory")

    def load_age_model(self):
        """Load age estimation model"""
        try:
            if os.path.exists(self.age_model_path):
                self.age_model = load_model(self.age_model_path)
                print("✅ Age estimation model loaded")
            else:
                print("⚠️ Age model not found. Using rule-based age estimation.")
                self.age_model = None
        except Exception as e:
            print(f"⚠️ Could not load age model: {e}")
            self.age_model = None

    def create_simple_age_model(self):
        """Create a simple age estimation model"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='linear')  # Age regression
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def eye_aspect_ratio(self, eye_landmarks):
        """Calculate eye aspect ratio for drowsiness detection"""
        # Vertical eye landmarks
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])

        # Horizontal eye landmark
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

        # Eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    def extract_eye_landmarks(self, landmarks):
        """Extract eye landmarks from 68-point facial landmarks"""
        # Left eye landmarks (points 42-47)
        left_eye = []
        for i in range(42, 48):
            left_eye.append([landmarks.part(i).x, landmarks.part(i).y])

        # Right eye landmarks (points 36-41)  
        right_eye = []
        for i in range(36, 42):
            right_eye.append([landmarks.part(i).x, landmarks.part(i).y])

        return np.array(left_eye), np.array(right_eye)

    def estimate_age_simple(self, face_region):
        """Simple rule-based age estimation"""
        # Convert to grayscale for analysis
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Calculate face texture features
        height, width = gray_face.shape

        # Simple heuristics based on face characteristics
        # This is a simplified approach - in practice, use trained models

        # Calculate edge density (wrinkles indicator)
        edges = cv2.Canny(gray_face, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)

        # Calculate contrast (skin texture indicator)
        contrast = np.std(gray_face)

        # Estimate age based on features
        if edge_density < 0.05 and contrast < 30:
            age = np.random.randint(18, 30)  # Young
        elif edge_density < 0.1 and contrast < 45:
            age = np.random.randint(30, 45)  # Middle-aged
        elif edge_density < 0.15 and contrast < 60:
            age = np.random.randint(45, 60)  # Mature
        else:
            age = np.random.randint(60, 80)  # Senior

        return age

    def estimate_age_ml(self, face_region):
        """ML-based age estimation"""
        if self.age_model is None:
            return self.estimate_age_simple(face_region)

        try:
            # Preprocess face for age model
            face_resized = cv2.resize(face_region, (64, 64))
            face_normalized = face_resized.astype('float32') / 255.0
            face_array = img_to_array(face_normalized)
            face_array = np.expand_dims(face_array, axis=0)

            # Predict age
            age_pred = self.age_model.predict(face_array, verbose=0)[0][0]
            age = max(1, min(100, int(age_pred)))  # Clamp between 1-100

            return age

        except Exception as e:
            print(f"Age estimation error: {e}")
            return self.estimate_age_simple(face_region)

    def detect_faces_multiple_methods(self, image):
        """Detect faces using multiple methods for better accuracy"""
        faces = []

        # Method 1: OpenCV Haar Cascades
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haar_faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        for (x, y, w, h) in haar_faces:
            faces.append({
                'bbox': (x, y, x+w, y+h),
                'method': 'haar',
                'confidence': 0.8  # Default confidence for Haar
            })

        # Method 2: face_recognition library
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)

            for (top, right, bottom, left) in face_locations:
                faces.append({
                    'bbox': (left, top, right, bottom),
                    'method': 'face_recognition',
                    'confidence': 0.9
                })
        except Exception as e:
            print(f"Face recognition error: {e}")

        # Method 3: MediaPipe
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)

            if results.detections:
                height, width = image.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * width)
                    y1 = int(bbox.ymin * height)
                    x2 = int((bbox.xmin + bbox.width) * width)
                    y2 = int((bbox.ymin + bbox.height) * height)

                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'method': 'mediapipe',
                        'confidence': detection.score[0]
                    })
        except Exception as e:
            print(f"MediaPipe error: {e}")

        # Remove duplicate faces (simple overlap check)
        unique_faces = self.remove_duplicate_faces(faces)
        return unique_faces

    def remove_duplicate_faces(self, faces, overlap_threshold=0.5):
        """Remove duplicate face detections"""
        if len(faces) <= 1:
            return faces

        unique_faces = []

        for face in faces:
            is_duplicate = False
            x1, y1, x2, y2 = face['bbox']

            for unique_face in unique_faces:
                ux1, uy1, ux2, uy2 = unique_face['bbox']

                # Calculate intersection over union (IoU)
                intersection_area = max(0, min(x2, ux2) - max(x1, ux1)) * max(0, min(y2, uy2) - max(y1, uy1))
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (ux2 - ux1) * (uy2 - uy1)
                union_area = area1 + area2 - intersection_area

                if union_area > 0:
                    iou = intersection_area / union_area
                    if iou > overlap_threshold:
                        is_duplicate = True
                        # Keep the one with higher confidence
                        if face['confidence'] > unique_face['confidence']:
                            unique_faces.remove(unique_face)
                            unique_faces.append(face)
                        break

            if not is_duplicate:
                unique_faces.append(face)

        return unique_faces

    def detect_drowsiness_in_image(self, image_path):
        """Detect drowsiness in a single image"""
        image = cv2.imread(image_path)
        if image is None:
            return None, [], 0, 0

        return self.process_frame_for_drowsiness(image)

    def process_frame_for_drowsiness(self, frame):
        """Process single frame for drowsiness detection"""
        # Detect faces
        faces = self.detect_faces_multiple_methods(frame)

        detections = []
        sleeping_count = 0
        total_people = len(faces)

        annotated_frame = frame.copy()

        for i, face_info in enumerate(faces):
            x1, y1, x2, y2 = face_info['bbox']

            # Extract face region
            face_region = frame[y1:y2, x1:x2]
            if face_region.size == 0:
                continue

            # Estimate age
            estimated_age = self.estimate_age_ml(face_region)

            # Detect drowsiness using eye analysis
            is_drowsy = self.detect_drowsiness_in_face(frame, (x1, y1, x2, y2))

            if is_drowsy:
                sleeping_count += 1

            # Create detection record
            detection = {
                'person_id': i + 1,
                'bbox': (x1, y1, x2, y2),
                'age': estimated_age,
                'is_sleeping': is_drowsy,
                'confidence': face_info['confidence'],
                'method': face_info['method']
            }
            detections.append(detection)

            # Draw annotations
            # Color: Red for sleeping, Green for awake
            color = (0, 0, 255) if is_drowsy else (0, 255, 0)
            thickness = 3 if is_drowsy else 2

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # Create label
            status = "SLEEPING" if is_drowsy else "AWAKE"
            label = f"Person {i+1}: {status}"
            age_label = f"Age: {estimated_age}"

            # Draw label background
            cv2.rectangle(annotated_frame, (x1, y1 - 50), (x2, y1), color, -1)

            # Draw text
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, age_label, (x1 + 5, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add summary to frame
        summary = f"Total: {total_people} | Sleeping: {sleeping_count}"
        cv2.putText(annotated_frame, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return annotated_frame, detections, total_people, sleeping_count

    def detect_drowsiness_in_face(self, frame, bbox):
        """Detect drowsiness in a specific face region"""
        x1, y1, x2, y2 = bbox

        if self.predictor is None:
            # Simple eye detection fallback
            return self.simple_eye_closure_detection(frame, bbox)

        # Convert to grayscale for dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create dlib rectangle
        rect = dlib.rectangle(x1, y1, x2, y2)

        # Get facial landmarks
        landmarks = self.predictor(gray, rect)

        # Extract eye landmarks
        left_eye, right_eye = self.extract_eye_landmarks(landmarks)

        # Calculate eye aspect ratios
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)

        # Average EAR
        ear = (left_ear + right_ear) / 2.0

        # Check if drowsy
        is_drowsy = ear < self.EYE_AR_THRESH

        return is_drowsy

    def simple_eye_closure_detection(self, frame, bbox):
        """Simple eye closure detection without landmarks"""
        x1, y1, x2, y2 = bbox
        face_region = frame[y1:y2, x1:x2]

        if face_region.size == 0:
            return False

        # Convert to grayscale
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Use eye cascade classifier
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 5, minSize=(10, 10))

        # If no eyes detected or very few, likely sleeping
        if len(eyes) < 2:
            return True

        # Additional check: analyze eye regions for closure
        eye_open_score = 0
        for (ex, ey, ew, eh) in eyes:
            eye_region = gray_face[ey:ey+eh, ex:ex+ew]
            if eye_region.size > 0:
                # Calculate variance (open eyes have more variation)
                variance = np.var(eye_region)
                if variance > 100:  # Threshold for open eyes
                    eye_open_score += 1

        # If less than 2 open eyes detected, consider drowsy
        is_drowsy = eye_open_score < 2

        return is_drowsy

    def process_video_for_drowsiness(self, video_path, output_path=None):
        """Process video for drowsiness detection"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None, 0, 0

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer if output path provided
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if output_path:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        max_sleeping = 0
        max_people = 0
        all_detections = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 3rd frame for performance
            if frame_count % 3 == 0:
                annotated_frame, detections, total_people, sleeping_count = self.process_frame_for_drowsiness(frame)

                max_people = max(max_people, total_people)
                max_sleeping = max(max_sleeping, sleeping_count)

                all_detections.extend(detections)

                if out:
                    out.write(annotated_frame)

                # Yield for real-time display (generator pattern)
                yield annotated_frame, detections, total_people, sleeping_count, frame_count, total_frames
            else:
                if out:
                    out.write(frame)

            frame_count += 1

        cap.release()
        if out:
            out.release()

        return output_path, max_people, max_sleeping

    def get_detection_summary(self, detections):
        """Generate detection summary"""
        if not detections:
            return {
                'total_people': 0,
                'sleeping_count': 0,
                'awake_count': 0,
                'average_age': 0,
                'sleeping_ages': [],
                'awake_ages': []
            }

        sleeping_people = [d for d in detections if d['is_sleeping']]
        awake_people = [d for d in detections if not d['is_sleeping']]

        sleeping_ages = [d['age'] for d in sleeping_people]
        awake_ages = [d['age'] for d in awake_people]
        all_ages = [d['age'] for d in detections]

        summary = {
            'total_people': len(detections),
            'sleeping_count': len(sleeping_people),
            'awake_count': len(awake_people),
            'average_age': np.mean(all_ages) if all_ages else 0,
            'sleeping_ages': sleeping_ages,
            'awake_ages': awake_ages
        }

        return summary

    def save_results(self, image_path, detections, output_dir='results'):
        """Save detection results"""
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Create results data
        summary = self.get_detection_summary(detections)

        results_data = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'detections': [],
            'summary': summary
        }

        for detection in detections:
            det_data = {
                'person_id': detection['person_id'],
                'bbox': detection['bbox'],
                'age': int(detection['age']),
                'is_sleeping': detection['is_sleeping'],
                'confidence': float(detection['confidence']),
                'detection_method': detection['method']
            }
            results_data['detections'].append(det_data)

        # Save to JSON
        json_path = os.path.join(output_dir, f"{base_name}_drowsiness_results.json")
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        return json_path
