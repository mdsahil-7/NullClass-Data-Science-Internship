import cv2
import numpy as np
import face_recognition
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import json
import pickle
from datetime import datetime
from colorthief import ColorThief
import webcolors
from scipy.spatial import KDTree
import dlib

class NationalityDetector:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir

        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        # Load models
        self.nationality_model = None
        self.emotion_model = None
        self.age_model = None

        # Load nationality classifications and color data
        self.nationality_classes = self.load_nationality_classes()
        self.emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.color_names = self.load_color_names()

        # Initialize models
        self.load_models()

        print("🌍 Nationality Detection System Initialized")

    def load_nationality_classes(self):
        """Load nationality classification classes"""
        # Based on major nationality datasets and demographics
        return {
            0: 'Indian',
            1: 'American', 
            2: 'African',
            3: 'Chinese',
            4: 'European',
            5: 'Middle Eastern',
            6: 'East Asian',
            7: 'Latino',
            8: 'Other'
        }

    def load_color_names(self):
        """Load color names mapping"""
        # Common dress colors with RGB values
        color_data = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'pink': (255, 192, 203),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'gray': (128, 128, 128),
            'brown': (165, 42, 42),
            'navy': (0, 0, 128),
            'maroon': (128, 0, 0),
            'olive': (128, 128, 0),
            'lime': (0, 255, 0),
            'aqua': (0, 255, 255),
            'teal': (0, 128, 128),
            'silver': (192, 192, 192),
            'gold': (255, 215, 0),
            'beige': (245, 245, 220)
        }
        return color_data

    def load_models(self):
        """Load pre-trained models"""
        try:
            # Load nationality model
            nationality_model_path = os.path.join(self.models_dir, 'nationality_model.h5')
            if os.path.exists(nationality_model_path):
                self.nationality_model = load_model(nationality_model_path)
                print("✅ Nationality model loaded")
            else:
                print("⚠️ Nationality model not found. Using rule-based detection.")

            # Load emotion model  
            emotion_model_path = os.path.join(self.models_dir, 'emotion_model.h5')
            if os.path.exists(emotion_model_path):
                self.emotion_model = load_model(emotion_model_path)
                print("✅ Emotion model loaded")
            else:
                print("⚠️ Emotion model not found. Using rule-based detection.")

            # Load age model
            age_model_path = os.path.join(self.models_dir, 'age_model.h5')
            if os.path.exists(age_model_path):
                self.age_model = load_model(age_model_path)
                print("✅ Age model loaded")
            else:
                print("⚠️ Age model not found. Using rule-based estimation.")

        except Exception as e:
            print(f"⚠️ Error loading models: {e}")

    def create_simple_models(self):
        """Create simple models for demonstration"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization

        # Nationality detection model
        nationality_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
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
            Dense(len(self.nationality_classes), activation='softmax')
        ])

        # Emotion detection model  
        emotion_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
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
            Dense(len(self.emotion_classes), activation='softmax')
        ])

        # Age estimation model
        age_model = Sequential([
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
            Dense(1, activation='linear')
        ])

        # Compile models
        nationality_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        age_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return nationality_model, emotion_model, age_model

    def detect_faces(self, image):
        """Detect faces in image using multiple methods"""
        faces = []

        # Method 1: OpenCV Haar Cascades
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haar_faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        for (x, y, w, h) in haar_faces:
            faces.append({
                'bbox': (x, y, x+w, y+h),
                'method': 'haar',
                'confidence': 0.8
            })

        # Method 2: MediaPipe
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

        return self.remove_duplicate_faces(faces)

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

                # Calculate IoU
                intersection_area = max(0, min(x2, ux2) - max(x1, ux1)) * max(0, min(y2, uy2) - max(y1, uy1))
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (ux2 - ux1) * (uy2 - uy1)
                union_area = area1 + area2 - intersection_area

                if union_area > 0:
                    iou = intersection_area / union_area
                    if iou > overlap_threshold:
                        is_duplicate = True
                        if face['confidence'] > unique_face['confidence']:
                            unique_faces.remove(unique_face)
                            unique_faces.append(face)
                        break

            if not is_duplicate:
                unique_faces.append(face)

        return unique_faces

    def predict_nationality_simple(self, face_region):
        """Simple rule-based nationality prediction"""
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)

        # Calculate skin tone features
        mean_hsv = np.mean(hsv, axis=(0, 1))
        mean_lab = np.mean(lab, axis=(0, 1))

        # Simple heuristics based on color analysis (simplified approach)
        # In practice, this would require trained models on diverse datasets
        h, s, v = mean_hsv
        l, a, b = mean_lab

        # Simplified classification based on skin tone analysis
        if l < 120 and a > 128:  # Darker skin tones
            return 'African' if np.random.random() > 0.5 else 'Indian'
        elif l > 160 and s < 50:  # Lighter skin tones  
            return 'American' if np.random.random() > 0.5 else 'European'
        elif a > 130 and b > 130:  # Warmer skin tones
            return 'Indian' if np.random.random() > 0.6 else 'Middle Eastern'
        elif h < 20 and s > 30:  # Asian characteristics
            return 'Chinese' if np.random.random() > 0.5 else 'East Asian'
        else:
            return np.random.choice(['Latino', 'Other'])

    def predict_nationality_ml(self, face_region):
        """ML-based nationality prediction"""
        if self.nationality_model is None:
            return self.predict_nationality_simple(face_region)

        try:
            # Preprocess face for nationality model
            face_resized = cv2.resize(face_region, (128, 128))
            face_normalized = face_resized.astype('float32') / 255.0
            face_array = np.expand_dims(face_normalized, axis=0)

            # Predict nationality
            predictions = self.nationality_model.predict(face_array, verbose=0)[0]
            nationality_id = np.argmax(predictions)
            confidence = predictions[nationality_id]

            nationality = self.nationality_classes.get(nationality_id, 'Other')

            return nationality, confidence

        except Exception as e:
            print(f"Nationality prediction error: {e}")
            return self.predict_nationality_simple(face_region)

    def predict_emotion_simple(self, face_region):
        """Simple rule-based emotion prediction"""
        # Convert to grayscale for edge analysis
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Calculate features
        edges = cv2.Canny(gray_face, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Calculate brightness and contrast
        brightness = np.mean(gray_face)
        contrast = np.std(gray_face)

        # Simple emotion heuristics (very simplified)
        if edge_density > 0.1 and contrast > 30:
            emotion = np.random.choice(['angry', 'surprise'], p=[0.6, 0.4])
        elif brightness > 150:
            emotion = np.random.choice(['happy', 'neutral'], p=[0.7, 0.3])
        elif brightness < 100:
            emotion = np.random.choice(['sad', 'fear'], p=[0.6, 0.4])
        else:
            emotion = np.random.choice(self.emotion_classes)

        return emotion

    def predict_emotion_ml(self, face_region):
        """ML-based emotion prediction"""
        if self.emotion_model is None:
            return self.predict_emotion_simple(face_region)

        try:
            # Preprocess face for emotion model
            face_resized = cv2.resize(face_region, (48, 48))
            face_normalized = face_resized.astype('float32') / 255.0
            face_array = np.expand_dims(face_normalized, axis=0)

            # Predict emotion
            predictions = self.emotion_model.predict(face_array, verbose=0)[0]
            emotion_id = np.argmax(predictions)
            confidence = predictions[emotion_id]

            emotion = self.emotion_classes[emotion_id]

            return emotion, confidence

        except Exception as e:
            print(f"Emotion prediction error: {e}")
            return self.predict_emotion_simple(face_region)

    def predict_age_simple(self, face_region):
        """Simple rule-based age estimation"""
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Calculate texture features
        edges = cv2.Canny(gray_face, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Calculate variance (smoothness)
        variance = np.var(gray_face)

        # Simple age estimation
        if edge_density < 0.05 and variance < 300:
            age = np.random.randint(18, 30)  # Young
        elif edge_density < 0.1 and variance < 500:
            age = np.random.randint(30, 45)  # Adult
        elif edge_density < 0.15:
            age = np.random.randint(45, 60)  # Middle-aged
        else:
            age = np.random.randint(60, 80)  # Senior

        return age

    def predict_age_ml(self, face_region):
        """ML-based age estimation"""
        if self.age_model is None:
            return self.predict_age_simple(face_region)

        try:
            # Preprocess face for age model
            face_resized = cv2.resize(face_region, (64, 64))
            face_normalized = face_resized.astype('float32') / 255.0
            face_array = np.expand_dims(face_normalized, axis=0)

            # Predict age
            age_pred = self.age_model.predict(face_array, verbose=0)[0][0]
            age = max(1, min(100, int(age_pred)))

            return age

        except Exception as e:
            print(f"Age prediction error: {e}")
            return self.predict_age_simple(face_region)

    def closest_color(self, rgb_color):
        """Find closest color name for RGB values"""
        min_distance = float('inf')
        closest_name = 'unknown'

        r, g, b = rgb_color

        for color_name, color_rgb in self.color_names.items():
            cr, cg, cb = color_rgb
            distance = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_name = color_name

        return closest_name

    def detect_dress_color(self, image, person_bbox):
        """Detect dominant dress/clothing color"""
        try:
            x1, y1, x2, y2 = person_bbox
            height = y2 - y1
            width = x2 - x1

            # Focus on lower part of detection (clothing area)
            clothing_y1 = y1 + int(height * 0.4)  # Skip face area
            clothing_y2 = min(y2 + int(height * 0.3), image.shape[0])  # Extend below face

            clothing_region = image[clothing_y1:clothing_y2, x1:x2]

            if clothing_region.size == 0:
                return 'unknown'

            # Method 1: Calculate dominant color
            # Reshape image to be a list of pixels
            pixels = clothing_region.reshape(-1, 3)

            # Remove very dark (shadows) and very light (overexposed) pixels
            mask = np.all([pixels[:, 0] > 30, pixels[:, 1] > 30, pixels[:, 2] > 30], axis=0)
            mask &= np.all([pixels[:, 0] < 220, pixels[:, 1] < 220, pixels[:, 2] < 220], axis=0)

            if np.sum(mask) > 0:
                filtered_pixels = pixels[mask]

                # Use k-means to find dominant color
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                kmeans.fit(filtered_pixels)

                # Get the most frequent cluster center
                labels = kmeans.labels_
                label_counts = np.bincount(labels)
                dominant_color = kmeans.cluster_centers_[np.argmax(label_counts)]

                # Convert BGR to RGB
                dominant_color_rgb = (int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0]))

                # Find closest color name
                color_name = self.closest_color(dominant_color_rgb)

                return color_name
            else:
                return 'unknown'

        except Exception as e:
            print(f"Dress color detection error: {e}")
            return 'unknown'

    def process_image(self, image_path):
        """Process image for nationality, emotion, age, and dress color detection"""
        image = cv2.imread(image_path)
        if image is None:
            return None, []

        # Detect faces
        faces = self.detect_faces(image)

        detections = []
        annotated_image = image.copy()

        for i, face_info in enumerate(faces):
            x1, y1, x2, y2 = face_info['bbox']

            # Extract face region
            face_region = image[y1:y2, x1:x2]
            if face_region.size == 0:
                continue

            # Predict nationality
            nationality_result = self.predict_nationality_ml(face_region)
            if isinstance(nationality_result, tuple):
                nationality, nat_confidence = nationality_result
            else:
                nationality = nationality_result
                nat_confidence = 0.8

            # Predict emotion
            emotion_result = self.predict_emotion_ml(face_region)
            if isinstance(emotion_result, tuple):
                emotion, emo_confidence = emotion_result
            else:
                emotion = emotion_result
                emo_confidence = 0.8

            # Initialize detection data
            detection = {
                'person_id': i + 1,
                'bbox': (x1, y1, x2, y2),
                'nationality': nationality,
                'nationality_confidence': nat_confidence,
                'emotion': emotion,
                'emotion_confidence': emo_confidence,
                'age': None,
                'dress_color': None
            }

            # Conditional predictions based on nationality
            if nationality == 'Indian':
                # Predict age and dress color
                detection['age'] = self.predict_age_ml(face_region)
                detection['dress_color'] = self.detect_dress_color(image, (x1, y1, x2, y2))

            elif nationality == 'American':
                # Predict only age
                detection['age'] = self.predict_age_ml(face_region)

            elif nationality == 'African':
                # Predict only dress color
                detection['dress_color'] = self.detect_dress_color(image, (x1, y1, x2, y2))

            # For other nationalities, only nationality and emotion are predicted (already done)

            detections.append(detection)

            # Draw annotations on image
            self.draw_annotations(annotated_image, detection)

        return annotated_image, detections

    def draw_annotations(self, image, detection):
        """Draw annotations on the image"""
        x1, y1, x2, y2 = detection['bbox']
        nationality = detection['nationality']
        emotion = detection['emotion']
        age = detection['age']
        dress_color = detection['dress_color']

        # Color coding by nationality
        nationality_colors = {
            'Indian': (0, 165, 255),      # Orange
            'American': (0, 0, 255),      # Red  
            'African': (0, 255, 0),       # Green
            'Chinese': (255, 255, 0),     # Yellow
            'European': (255, 0, 255),    # Magenta
            'Middle Eastern': (128, 0, 128), # Purple
            'East Asian': (255, 165, 0),  # Light Blue
            'Latino': (0, 255, 255),      # Cyan
            'Other': (128, 128, 128)      # Gray
        }

        color = nationality_colors.get(nationality, (128, 128, 128))

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        # Prepare label text
        label_lines = [
            f"Person {detection['person_id']}",
            f"Nation: {nationality}",
            f"Emotion: {emotion}"
        ]

        if age is not None:
            label_lines.append(f"Age: {age}")

        if dress_color is not None:
            label_lines.append(f"Dress: {dress_color}")

        # Draw label background and text
        line_height = 20
        total_height = len(label_lines) * line_height + 10

        cv2.rectangle(image, (x1, y1 - total_height), (x2, y1), color, -1)

        for i, line in enumerate(label_lines):
            y_text = y1 - total_height + 15 + (i * line_height)
            cv2.putText(image, line, (x1 + 5, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def get_detection_summary(self, detections):
        """Generate detection summary"""
        if not detections:
            return {
                'total_people': 0,
                'nationalities': {},
                'emotions': {},
                'ages': [],
                'dress_colors': {}
            }

        nationalities = {}
        emotions = {}
        ages = []
        dress_colors = {}

        for detection in detections:
            # Count nationalities
            nat = detection['nationality']
            nationalities[nat] = nationalities.get(nat, 0) + 1

            # Count emotions
            emo = detection['emotion']
            emotions[emo] = emotions.get(emo, 0) + 1

            # Collect ages
            if detection['age'] is not None:
                ages.append(detection['age'])

            # Count dress colors
            if detection['dress_color'] is not None:
                color = detection['dress_color']
                dress_colors[color] = dress_colors.get(color, 0) + 1

        return {
            'total_people': len(detections),
            'nationalities': nationalities,
            'emotions': emotions,
            'ages': ages,
            'dress_colors': dress_colors,
            'average_age': np.mean(ages) if ages else None
        }

    def save_results(self, image_path, detections, output_dir='results'):
        """Save detection results"""
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
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
                'nationality': detection['nationality'],
                'nationality_confidence': float(detection['nationality_confidence']),
                'emotion': detection['emotion'],
                'emotion_confidence': float(detection['emotion_confidence']),
                'age': int(detection['age']) if detection['age'] is not None else None,
                'dress_color': detection['dress_color']
            }
            results_data['detections'].append(det_data)

        json_path = os.path.join(output_dir, f"{base_name}_nationality_results.json")
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        return json_path
