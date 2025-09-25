import cv2
import pandas as pd
import numpy as np
from datetime import datetime, time
import os
import pickle
from face_recognition import FaceRecognizer
from emotion_detection import EmotionDetector

class AttendanceSystem:
    def __init__(self, student_list=None, output_file='attendance_records.csv'):
        self.face_recognizer = FaceRecognizer()
        self.emotion_detector = EmotionDetector()
        self.output_file = output_file
        self.attendance_records = []

        # Default student list if none provided
        if student_list is None:
            self.student_list = ['Student_1', 'Student_2', 'Student_3', 'Student_4', 'Student_5']
        else:
            self.student_list = student_list

        # Initialize attendance tracking
        self.present_students = set()
        self.session_active = False

        self.create_output_file()

    def create_output_file(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.output_file):
            df = pd.DataFrame(columns=[
                'Date', 'Time', 'Student_Name', 'Status', 'Emotion', 
                'Emotion_Confidence', 'Face_Confidence', 'Session_Duration'
            ])
            df.to_csv(self.output_file, index=False)
            print(f"Created attendance file: {self.output_file}")

    def is_attendance_time(self):
        """Check if current time is within attendance window (9:30 AM - 10:00 AM)"""
        current_time = datetime.now().time()
        start_time = time(9, 30)  # 9:30 AM
        end_time = time(10, 0)    # 10:00 AM
        return start_time <= current_time <= end_time

    def mark_attendance(self, student_name, emotion, emotion_confidence, face_confidence):
        """Mark attendance for a student"""
        current_datetime = datetime.now()

        # Check if student is already marked present in this session
        if student_name in self.present_students:
            return

        # Add to present students
        self.present_students.add(student_name)

        # Create attendance record
        record = {
            'Date': current_datetime.strftime('%Y-%m-%d'),
            'Time': current_datetime.strftime('%H:%M:%S'),
            'Student_Name': student_name,
            'Status': 'Present',
            'Emotion': emotion,
            'Emotion_Confidence': round(emotion_confidence, 3),
            'Face_Confidence': round(face_confidence, 3),
            'Session_Duration': '30_minutes'
        }

        self.attendance_records.append(record)
        print(f"✓ Marked {student_name} as Present - Emotion: {emotion} ({emotion_confidence:.3f})")

    def save_attendance_records(self):
        """Save attendance records to CSV"""
        if self.attendance_records:
            df_new = pd.DataFrame(self.attendance_records)

            # Load existing records if file exists
            if os.path.exists(self.output_file):
                df_existing = pd.read_csv(self.output_file)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new

            # Save to CSV
            df_combined.to_csv(self.output_file, index=False)
            print(f"Saved {len(self.attendance_records)} attendance records to {self.output_file}")

            # Also save as Excel
            excel_file = self.output_file.replace('.csv', '.xlsx')
            df_combined.to_excel(excel_file, index=False)
            print(f"Also saved as Excel: {excel_file}")

    def mark_absent_students(self):
        """Mark students who were not detected as absent"""
        current_datetime = datetime.now()

        for student in self.student_list:
            if student not in self.present_students:
                record = {
                    'Date': current_datetime.strftime('%Y-%m-%d'),
                    'Time': current_datetime.strftime('%H:%M:%S'),
                    'Student_Name': student,
                    'Status': 'Absent',
                    'Emotion': 'N/A',
                    'Emotion_Confidence': 0.0,
                    'Face_Confidence': 0.0,
                    'Session_Duration': '30_minutes'
                }
                self.attendance_records.append(record)
                print(f"✗ Marked {student} as Absent")

    def display_attendance_summary(self):
        """Display attendance summary"""
        print("\n" + "="*50)
        print("ATTENDANCE SUMMARY")
        print("="*50)
        print(f"Session Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Session Time: 9:30 AM - 10:00 AM")
        print(f"Total Students: {len(self.student_list)}")
        print(f"Present: {len(self.present_students)}")
        print(f"Absent: {len(self.student_list) - len(self.present_students)}")
        print(f"Attendance Rate: {len(self.present_students)/len(self.student_list)*100:.1f}%")
        print("\nPresent Students:")
        for student in sorted(self.present_students):
            print(f"  ✓ {student}")
        print("\nAbsent Students:")
        absent_students = set(self.student_list) - self.present_students
        for student in sorted(absent_students):
            print(f"  ✗ {student}")
        print("="*50)

    def run_attendance_session(self, camera_index=0, display_video=True):
        """Run the attendance session"""
        print("Starting Attendance System...")
        print(f"Active Time Window: 9:30 AM - 10:00 AM")
        print(f"Current Time: {datetime.now().strftime('%H:%M:%S')}")

        # Check if it's attendance time
        if not self.is_attendance_time():
            print("❌ Current time is outside attendance window!")
            print("Please run the system between 9:30 AM and 10:00 AM")
            return

        print("✅ Attendance session is active!")
        self.session_active = True

        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return

        print("📷 Camera initialized. Press 'q' to quit")

        frame_count = 0
        process_every_n_frames = 5  # Process every 5th frame for performance

        try:
            while self.is_attendance_time() and self.session_active:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break

                frame_count += 1

                # Process every nth frame to improve performance
                if frame_count % process_every_n_frames == 0:
                    # Face recognition
                    recognized_faces = self.face_recognizer.recognize_faces(frame)

                    # Emotion detection
                    emotions = self.emotion_detector.detect_emotions(frame)

                    # Process recognized faces
                    for face in recognized_faces:
                        if face['name'] != "Unknown" and face['confidence'] > 0.6:
                            # Find corresponding emotion
                            dominant_emotion, emotion_conf = self.emotion_detector.get_dominant_emotion(emotions)

                            # Mark attendance
                            self.mark_attendance(
                                face['name'], 
                                dominant_emotion, 
                                emotion_conf, 
                                face['confidence']
                            )

                # Draw visualizations
                if display_video:
                    # Draw face boxes
                    frame = self.face_recognizer.draw_face_boxes(frame, recognized_faces)

                    # Draw emotion labels
                    if frame_count % process_every_n_frames == 0:
                        frame = self.emotion_detector.draw_emotion_labels(frame, emotions)

                    # Add time and status information
                    current_time = datetime.now().strftime('%H:%M:%S')
                    cv2.putText(frame, f"Time: {current_time}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Present: {len(self.present_students)}/{len(self.student_list)}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Show frame
                    cv2.imshow('Student Attendance System', frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n🛑 Session stopped by user")
                    break

            if not self.is_attendance_time():
                print("\n⏰ Attendance window closed")

        except KeyboardInterrupt:
            print("\n🛑 Session interrupted by user")

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()

            # Mark absent students and save records
            self.mark_absent_students()
            self.save_attendance_records()
            self.display_attendance_summary()

            print("\n✅ Attendance session completed!")

    def train_face_recognition(self, dataset_path):
        """Train the face recognition model"""
        print("Training face recognition model...")
        self.face_recognizer.train_from_folder(dataset_path)
        print("Face recognition training completed!")

# Demo/Testing functions
def create_sample_data():
    """Create sample training data structure"""
    print("Creating sample dataset structure...")

    # Create directories
    dataset_dir = "student_dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    students = ['Student_1', 'Student_2', 'Student_3', 'Student_4', 'Student_5']

    for student in students:
        student_dir = os.path.join(dataset_dir, student)
        os.makedirs(student_dir, exist_ok=True)

    print(f"Created dataset structure in {dataset_dir}/")
    print("Please add student photos to respective folders before training!")
    return dataset_dir

if __name__ == "__main__":
    # Create sample dataset structure
    dataset_path = create_sample_data()

    # Initialize attendance system
    attendance_sys = AttendanceSystem()

    print("\nAttendance System Ready!")
    print("\nOptions:")
    print("1. Train face recognition (add photos to student_dataset/ folders first)")
    print("2. Run attendance session")
    print("\nFor demo purposes, you can run without training, but results will be limited.")

    # Uncomment the following lines to run:
    # attendance_sys.train_face_recognition(dataset_path)  # Run this first
    # attendance_sys.run_attendance_session()  # Then run this
