import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import webcolors
import os
import json
import pickle
from datetime import datetime
from collections import Counter
import imutils

class CarColorDetector:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir

        # Initialize YOLO model for object detection
        self.yolo_model = None
        self.load_yolo_model()

        # COCO class IDs for vehicles and people
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }

        self.person_class = 0  # COCO class ID for person

        # Color definitions for car color detection
        self.color_ranges = {
            'blue': {
                'lower': np.array([100, 50, 50]),
                'upper': np.array([130, 255, 255])
            },
            'red': {
                'lower': np.array([0, 50, 50]),
                'upper': np.array([10, 255, 255])
            },
            'green': {
                'lower': np.array([35, 50, 50]),
                'upper': np.array([85, 255, 255])
            },
            'yellow': {
                'lower': np.array([15, 50, 50]),
                'upper': np.array([35, 255, 255])
            },
            'white': {
                'lower': np.array([0, 0, 200]),
                'upper': np.array([180, 30, 255])
            },
            'black': {
                'lower': np.array([0, 0, 0]),
                'upper': np.array([180, 255, 50])
            },
            'gray': {
                'lower': np.array([0, 0, 50]),
                'upper': np.array([180, 30, 200])
            }
        }

        print("🚗 Car Color Detection System Initialized")

    def load_yolo_model(self):
        """Load YOLO model for object detection"""
        try:
            # Try to load custom model first
            model_path = os.path.join(self.models_dir, 'yolov8n.pt')
            if os.path.exists(model_path):
                self.yolo_model = YOLO(model_path)
                print("✅ Custom YOLO model loaded")
            else:
                # Load pre-trained YOLOv8 model
                self.yolo_model = YOLO('yolov8n.pt')  # Will download automatically
                print("✅ Pre-trained YOLO model loaded")

        except Exception as e:
            print(f"⚠️ Error loading YOLO model: {e}")
            print("Using fallback detection methods...")
            self.yolo_model = None

    def detect_objects(self, image, confidence_threshold=0.5):
        """Detect cars and people in image using YOLO"""
        if self.yolo_model is None:
            return self.fallback_detection(image)

        try:
            # Run YOLO inference
            results = self.yolo_model(image, conf=confidence_threshold, verbose=False)

            cars = []
            people = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])

                        # Check if it's a vehicle
                        if class_id in self.vehicle_classes:
                            cars.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self.vehicle_classes[class_id]
                            })

                        # Check if it's a person
                        elif class_id == self.person_class:
                            people.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': 'person'
                            })

            return cars, people

        except Exception as e:
            print(f"YOLO detection error: {e}")
            return self.fallback_detection(image)

    def fallback_detection(self, image):
        """Fallback detection using OpenCV cascade classifiers"""
        cars = []
        people = []

        try:
            # Load cascade classifiers
            car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
            pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect cars
            if os.path.exists(cv2.data.haarcascades + 'haarcascade_car.xml'):
                car_detections = car_cascade.detectMultiScale(gray, 1.1, 3, minSize=(50, 50))
                for (x, y, w, h) in car_detections:
                    cars.append({
                        'bbox': (x, y, x+w, y+h),
                        'confidence': 0.7,
                        'class_id': 2,
                        'class_name': 'car'
                    })

            # Detect people
            people_detections = pedestrian_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 60))
            for (x, y, w, h) in people_detections:
                people.append({
                    'bbox': (x, y, x+w, y+h),
                    'confidence': 0.6,
                    'class_id': 0,
                    'class_name': 'person'
                })

        except Exception as e:
            print(f"Fallback detection error: {e}")

        return cars, people

    def extract_dominant_color(self, image_region, k=3):
        """Extract dominant color from image region using K-means clustering"""
        try:
            if image_region.size == 0:
                return 'unknown'

            # Reshape image to list of pixels
            pixels = image_region.reshape(-1, 3)

            # Remove very dark and very light pixels (shadows and reflections)
            mask = np.all([pixels[:, 0] > 30, pixels[:, 1] > 30, pixels[:, 2] > 30], axis=0)
            mask &= np.all([pixels[:, 0] < 220, pixels[:, 1] < 220, pixels[:, 2] < 220], axis=0)

            if np.sum(mask) < 10:  # Not enough valid pixels
                return 'unknown'

            filtered_pixels = pixels[mask]

            # Apply K-means clustering
            kmeans = KMeans(n_clusters=min(k, len(filtered_pixels)), random_state=42, n_init=10)
            kmeans.fit(filtered_pixels)

            # Get most frequent cluster
            labels = kmeans.labels_
            label_counts = np.bincount(labels)
            dominant_color = kmeans.cluster_centers_[np.argmax(label_counts)]

            return dominant_color.astype(int)

        except Exception as e:
            print(f"Color extraction error: {e}")
            return 'unknown'

    def classify_color(self, dominant_color):
        """Classify dominant color into named categories"""
        if isinstance(dominant_color, str):
            return dominant_color

        try:
            # Convert BGR to HSV for better color classification
            bgr_color = np.uint8([[dominant_color]])
            hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]

            # Check each color range
            for color_name, color_range in self.color_ranges.items():
                lower = color_range['lower']
                upper = color_range['upper']

                # Special handling for red (wraps around hue)
                if color_name == 'red':
                    if (hsv_color[0] >= 0 and hsv_color[0] <= 10) or (hsv_color[0] >= 170):
                        if hsv_color[1] >= 50 and hsv_color[2] >= 50:
                            return 'red'
                else:
                    if (lower[0] <= hsv_color[0] <= upper[0] and 
                        lower[1] <= hsv_color[1] <= upper[1] and 
                        lower[2] <= hsv_color[2] <= upper[2]):
                        return color_name

            # If no match found, return the closest named color
            return self.closest_color_name(dominant_color)

        except Exception as e:
            print(f"Color classification error: {e}")
            return 'unknown'

    def closest_color_name(self, rgb_color):
        """Find closest named color using webcolors library"""
        try:
            # Convert BGR to RGB
            r, g, b = int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0])

            min_colors = {}
            for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(key)
                rd = (r_c - r) ** 2
                gd = (g_c - g) ** 2
                bd = (b_c - b) ** 2
                min_colors[(rd + gd + bd)] = name

            closest_color = min_colors[min(min_colors.keys())]

            # Map to our simplified color categories
            color_mapping = {
                'blue': ['blue', 'navy', 'darkblue', 'mediumblue', 'royalblue', 'steelblue', 'lightblue'],
                'red': ['red', 'darkred', 'crimson', 'maroon', 'tomato', 'coral'],
                'green': ['green', 'darkgreen', 'forestgreen', 'lime', 'olive'],
                'yellow': ['yellow', 'gold', 'orange', 'darkorange'],
                'white': ['white', 'ivory', 'snow', 'whitesmoke'],
                'black': ['black', 'darkslategray', 'dimgray'],
                'gray': ['gray', 'silver', 'lightgray', 'darkgray']
            }

            for category, colors in color_mapping.items():
                if closest_color.lower() in colors:
                    return category

            return closest_color.lower()

        except Exception as e:
            print(f"Closest color error: {e}")
            return 'unknown'

    def detect_car_color(self, image, car_bbox):
        """Detect the color of a specific car"""
        x1, y1, x2, y2 = car_bbox

        # Extract car region
        car_region = image[y1:y2, x1:x2]

        if car_region.size == 0:
            return 'unknown'

        # Focus on the main body of the car (avoid windows, tires, etc.)
        h, w = car_region.shape[:2]

        # Extract middle portion of the car (main body)
        body_y1 = int(h * 0.2)
        body_y2 = int(h * 0.7)
        body_x1 = int(w * 0.1)
        body_x2 = int(w * 0.9)

        car_body = car_region[body_y1:body_y2, body_x1:body_x2]

        if car_body.size == 0:
            car_body = car_region

        # Extract dominant color
        dominant_color = self.extract_dominant_color(car_body)

        # Classify color
        color_name = self.classify_color(dominant_color)

        return color_name

    def process_traffic_image(self, image_path):
        """Process traffic image for car color detection and counting"""
        image = cv2.imread(image_path)
        if image is None:
            return None, [], 0, 0, 0

        return self.process_traffic_frame(image)

    def process_traffic_frame(self, frame):
        """Process single frame for traffic analysis"""
        # Detect objects
        cars, people = self.detect_objects(frame)

        # Analyze car colors
        car_detections = []
        blue_car_count = 0
        other_car_count = 0

        annotated_frame = frame.copy()

        # Process each car
        for i, car in enumerate(cars):
            bbox = car['bbox']
            x1, y1, x2, y2 = bbox

            # Detect car color
            car_color = self.detect_car_color(frame, bbox)

            # Create detection record
            detection = {
                'car_id': i + 1,
                'bbox': bbox,
                'color': car_color,
                'confidence': car['confidence'],
                'class_name': car['class_name']
            }
            car_detections.append(detection)

            # Count blue vs other cars
            if car_color.lower() == 'blue':
                blue_car_count += 1
                # Red rectangle for blue cars
                color = (0, 0, 255)  # Red in BGR
                label = f"Blue Car {i+1}"
            else:
                other_car_count += 1
                # Blue rectangle for other color cars
                color = (255, 0, 0)  # Blue in BGR
                label = f"{car_color.title()} Car {i+1}"

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

            # Draw label
            label_full = f"{label} ({car['confidence']:.2f})"

            # Label background
            (label_width, label_height), _ = cv2.getTextSize(label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), color, -1)

            # Label text
            cv2.putText(annotated_frame, label_full, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Process people
        people_detections = []
        for i, person in enumerate(people):
            bbox = person['bbox']
            x1, y1, x2, y2 = bbox

            detection = {
                'person_id': i + 1,
                'bbox': bbox,
                'confidence': person['confidence']
            }
            people_detections.append(detection)

            # Draw green bounding box for people
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"Person {i+1} ({person['confidence']:.2f})"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Add summary information
        total_cars = blue_car_count + other_car_count
        total_people = len(people_detections)

        # Summary text
        summary_lines = [
            f"Total Cars: {total_cars}",
            f"Blue Cars: {blue_car_count}",
            f"Other Cars: {other_car_count}", 
            f"People: {total_people}"
        ]

        # Draw summary
        y_offset = 30
        for line in summary_lines:
            cv2.putText(annotated_frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated_frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            y_offset += 30

        return annotated_frame, car_detections, people_detections, blue_car_count, other_car_count

    def get_detection_summary(self, car_detections, people_detections):
        """Generate detection summary statistics"""
        if not car_detections and not people_detections:
            return {
                'total_cars': 0,
                'blue_cars': 0,
                'other_cars': 0,
                'total_people': 0,
                'car_colors': {},
                'average_confidence': 0.0
            }

        # Car statistics
        car_colors = {}
        blue_cars = 0
        other_cars = 0
        all_confidences = []

        for car in car_detections:
            color = car['color']
            car_colors[color] = car_colors.get(color, 0) + 1
            all_confidences.append(car['confidence'])

            if color.lower() == 'blue':
                blue_cars += 1
            else:
                other_cars += 1

        # Add people confidences
        for person in people_detections:
            all_confidences.append(person['confidence'])

        return {
            'total_cars': len(car_detections),
            'blue_cars': blue_cars,
            'other_cars': other_cars,
            'total_people': len(people_detections),
            'car_colors': car_colors,
            'average_confidence': np.mean(all_confidences) if all_confidences else 0.0
        }

    def save_results(self, image_path, car_detections, people_detections, output_dir='results'):
        """Save detection results"""
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        summary = self.get_detection_summary(car_detections, people_detections)

        results_data = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'car_detections': [],
            'people_detections': [],
            'summary': summary
        }

        # Car detections
        for car in car_detections:
            car_data = {
                'car_id': car['car_id'],
                'bbox': car['bbox'],
                'color': car['color'],
                'confidence': float(car['confidence']),
                'class_name': car['class_name']
            }
            results_data['car_detections'].append(car_data)

        # People detections  
        for person in people_detections:
            person_data = {
                'person_id': person['person_id'],
                'bbox': person['bbox'],
                'confidence': float(person['confidence'])
            }
            results_data['people_detections'].append(person_data)

        json_path = os.path.join(output_dir, f"{base_name}_traffic_analysis.json")
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        return json_path
