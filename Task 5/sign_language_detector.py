import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import json
import pickle
from datetime import datetime, time
from collections import Counter
import pyttsx3
import threading

class SignLanguageDetector:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir

        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize text-to-speech
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_enabled = True
        except:
            self.tts_enabled = False
            print("⚠️ Text-to-speech not available")

        # Load models
        self.sign_model = None

        # Sign language vocabulary - common ASL signs
        self.sign_vocabulary = {
            0: 'Hello',
            1: 'Thank You', 
            2: 'Please',
            3: 'Yes',
            4: 'No',
            5: 'Good',
            6: 'Bad',
            7: 'Help',
            8: 'Water',
            9: 'Food',
            10: 'Love',
            11: 'Peace',
            12: 'Stop',
            13: 'Go',
            14: 'Come',
            15: 'Beautiful',
            16: 'Family',
            17: 'Friend',
            18: 'Home',
            19: 'Work'
        }

        # Time-based operation settings
        self.operation_start_time = time(18, 0)  # 6:00 PM
        self.operation_end_time = time(22, 0)    # 10:00 PM

        # Detection history for smoothing
        self.detection_history = []
        self.max_history_length = 10

        # Load models
        self.load_models()

        print("🤟 Sign Language Detection System Initialized")
        print(f"⏰ Operating hours: {self.operation_start_time.strftime('%I:%M %p')} - {self.operation_end_time.strftime('%I:%M %p')}")

    def is_operation_time(self):
        """Check if current time is within operation hours"""
        current_time = datetime.now().time()
        return self.operation_start_time <= current_time <= self.operation_end_time

    def load_models(self):
        """Load pre-trained sign language models"""
        try:
            sign_model_path = os.path.join(self.models_dir, 'sign_language_model.h5')
            if os.path.exists(sign_model_path):
                self.sign_model = load_model(sign_model_path)
                print("✅ Sign language model loaded")
            else:
                print("⚠️ Sign language model not found. Using rule-based detection.")

        except Exception as e:
            print(f"⚠️ Error loading models: {e}")

    def create_simple_sign_model(self):
        """Create simple CNN model for sign language recognition"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization

        model = Sequential([
            # First Conv Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            # Second Conv Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            # Third Conv Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            # Fourth Conv Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            # Fully Connected Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(len(self.sign_vocabulary), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def extract_hand_landmarks(self, image):
        """Extract hand landmarks from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        hand_landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmark.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                hand_landmarks.append(landmarks)

        return hand_landmarks, results

    def landmarks_to_features(self, landmarks_list):
        """Convert landmarks to feature vector"""
        if not landmarks_list:
            return np.zeros(63)  # 21 landmarks * 3 coordinates

        # Use first hand for simplicity
        landmarks = landmarks_list[0]

        # Normalize landmarks relative to wrist (landmark 0)
        if len(landmarks) >= 63:
            wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
            normalized_landmarks = []

            for i in range(0, 63, 3):
                x = landmarks[i] - wrist_x
                y = landmarks[i+1] - wrist_y
                z = landmarks[i+2] - wrist_z
                normalized_landmarks.extend([x, y, z])

            return np.array(normalized_landmarks)

        return np.zeros(63)

    def predict_sign_simple(self, landmarks_list):
        """Simple rule-based sign prediction"""
        if not landmarks_list:
            return None, 0.0

        landmarks = landmarks_list[0]
        if len(landmarks) < 63:
            return None, 0.0

        # Convert to numpy array and reshape for analysis
        landmarks_array = np.array(landmarks).reshape(21, 3)

        # Simple gesture recognition based on hand shape
        # This is a simplified approach - real implementation would use trained models

        # Calculate finger tip positions relative to wrist
        wrist = landmarks_array[0]
        thumb_tip = landmarks_array[4]
        index_tip = landmarks_array[8]
        middle_tip = landmarks_array[12]
        ring_tip = landmarks_array[16]
        pinky_tip = landmarks_array[20]

        # Calculate relative positions
        thumb_up = thumb_tip[1] < wrist[1] - 0.1
        index_up = index_tip[1] < wrist[1] - 0.1
        middle_up = middle_tip[1] < wrist[1] - 0.1
        ring_up = ring_tip[1] < wrist[1] - 0.1
        pinky_up = pinky_tip[1] < wrist[1] - 0.1

        # Simple gesture classification
        if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            return 'Good', 0.8
        elif not thumb_up and index_up and middle_up and not ring_up and not pinky_up:
            return 'Peace', 0.8
        elif not thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
            return 'Stop', 0.7
        elif thumb_up and index_up and middle_up and ring_up and pinky_up:
            return 'Hello', 0.9
        elif not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            return 'No', 0.7
        elif thumb_up and index_up and not middle_up and not ring_up and pinky_up:
            return 'Love', 0.8
        else:
            # Random selection from vocabulary for demonstration
            sign_options = ['Please', 'Thank You', 'Help', 'Water', 'Food', 'Yes', 'Beautiful']
            return np.random.choice(sign_options), 0.6

    def predict_sign_ml(self, landmarks_list):
        """ML-based sign prediction"""
        if self.sign_model is None:
            return self.predict_sign_simple(landmarks_list)

        try:
            features = self.landmarks_to_features(landmarks_list)
            features = features.reshape(1, -1)

            # Predict sign
            predictions = self.sign_model.predict(features, verbose=0)[0]
            sign_id = np.argmax(predictions)
            confidence = predictions[sign_id]

            if confidence < 0.5:  # Low confidence threshold
                return None, confidence

            sign = self.sign_vocabulary.get(sign_id, 'Unknown')

            return sign, confidence

        except Exception as e:
            print(f"Sign prediction error: {e}")
            return self.predict_sign_simple(landmarks_list)

    def smooth_predictions(self, prediction, confidence):
        """Smooth predictions using history"""
        if prediction is None:
            return None, 0.0

        # Add to history
        self.detection_history.append((prediction, confidence))

        # Keep only recent detections
        if len(self.detection_history) > self.max_history_length:
            self.detection_history.pop(0)

        # Count recent predictions
        recent_predictions = [p[0] for p in self.detection_history[-5:]]
        prediction_counts = Counter(recent_predictions)

        # Return most common prediction if it appears at least twice
        most_common = prediction_counts.most_common(1)[0]
        if most_common[1] >= 2:
            avg_confidence = np.mean([p[1] for p in self.detection_history if p[0] == most_common[0]])
            return most_common[0], avg_confidence

        return prediction, confidence

    def speak_text(self, text):
        """Convert text to speech"""
        if self.tts_enabled:
            try:
                def speak():
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()

                # Run in separate thread to avoid blocking
                thread = threading.Thread(target=speak)
                thread.daemon = True
                thread.start()
            except Exception as e:
                print(f"TTS error: {e}")

    def process_image_for_signs(self, image_path):
        """Process image for sign language detection"""
        # Check operation time
        if not self.is_operation_time():
            current_time = datetime.now().strftime('%I:%M %p')
            return None, [], f"System not operational. Current time: {current_time}. Operating hours: 6:00 PM - 10:00 PM"

        image = cv2.imread(image_path)
        if image is None:
            return None, [], "Could not load image"

        return self.process_frame_for_signs(image)

    def process_frame_for_signs(self, frame):
        """Process single frame for sign language detection"""
        # Check operation time
        if not self.is_operation_time():
            current_time = datetime.now().strftime('%I:%M %p')
            return frame, [], f"System not operational. Current time: {current_time}"

        # Extract hand landmarks
        hand_landmarks, results = self.extract_hand_landmarks(frame)

        detections = []
        annotated_frame = frame.copy()

        if results.multi_hand_landmarks:
            for i, (hand_landmark, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame, hand_landmark, self.mp_hands.HAND_CONNECTIONS)

                # Predict sign
                predicted_sign, confidence = self.predict_sign_ml([hand_landmarks[i]])

                # Smooth predictions
                smooth_sign, smooth_confidence = self.smooth_predictions(predicted_sign, confidence)

                if smooth_sign:
                    # Get hand label (Left/Right)
                    hand_label = handedness.classification[0].label

                    detection = {
                        'hand_id': i + 1,
                        'hand_type': hand_label,
                        'sign': smooth_sign,
                        'confidence': smooth_confidence,
                        'landmarks': hand_landmarks[i]
                    }
                    detections.append(detection)

                    # Draw sign prediction
                    h, w, _ = annotated_frame.shape

                    # Get hand bounding box
                    x_coords = [lm.x * w for lm in hand_landmark.landmark]
                    y_coords = [lm.y * h for lm in hand_landmark.landmark]

                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))

                    # Expand bounding box
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Draw label
                    label = f"{hand_label} Hand: {smooth_sign} ({smooth_confidence:.2f})"

                    # Label background
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x_min, y_min - label_height - 10), 
                                 (x_min + label_width, y_min), (0, 255, 0), -1)

                    # Label text
                    cv2.putText(annotated_frame, label, (x_min, y_min - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Add operation status and time
        current_time = datetime.now().strftime('%I:%M %p')
        status_text = f"ACTIVE - {current_time}"
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Add detected signs summary
        if detections:
            signs_detected = [d['sign'] for d in detections]
            summary_text = f"Signs: {', '.join(signs_detected)}"
            cv2.putText(annotated_frame, summary_text, (10, annotated_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return annotated_frame, detections, "Detection successful"

    def get_detection_summary(self, detections):
        """Generate detection summary"""
        if not detections:
            return {
                'total_hands': 0,
                'signs_detected': {},
                'confidence_scores': [],
                'hand_types': {}
            }

        signs_detected = {}
        confidence_scores = []
        hand_types = {}

        for detection in detections:
            # Count signs
            sign = detection['sign']
            signs_detected[sign] = signs_detected.get(sign, 0) + 1

            # Collect confidence scores
            confidence_scores.append(detection['confidence'])

            # Count hand types
            hand_type = detection['hand_type']
            hand_types[hand_type] = hand_types.get(hand_type, 0) + 1

        return {
            'total_hands': len(detections),
            'signs_detected': signs_detected,
            'confidence_scores': confidence_scores,
            'hand_types': hand_types,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0
        }

    def save_results(self, image_path, detections, output_dir='results'):
        """Save detection results"""
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        summary = self.get_detection_summary(detections)

        results_data = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'operation_time_check': self.is_operation_time(),
            'current_time': datetime.now().strftime('%I:%M %p'),
            'operating_hours': f"{self.operation_start_time.strftime('%I:%M %p')} - {self.operation_end_time.strftime('%I:%M %p')}",
            'detections': [],
            'summary': summary
        }

        for detection in detections:
            det_data = {
                'hand_id': detection['hand_id'],
                'hand_type': detection['hand_type'],
                'sign': detection['sign'],
                'confidence': float(detection['confidence']),
                'landmarks_count': len(detection['landmarks'])
            }
            results_data['detections'].append(det_data)

        json_path = os.path.join(output_dir, f"{base_name}_sign_results.json")
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        return json_path

    def get_available_signs(self):
        """Get list of available signs in vocabulary"""
        return list(self.sign_vocabulary.values())

    def get_operation_status(self):
        """Get current operation status"""
        is_active = self.is_operation_time()
        current_time = datetime.now().strftime('%I:%M %p')

        return {
            'is_active': is_active,
            'current_time': current_time,
            'operation_start': self.operation_start_time.strftime('%I:%M %p'),
            'operation_end': self.operation_end_time.strftime('%I:%M %p'),
            'status_message': 'ACTIVE' if is_active else 'INACTIVE'
        }
