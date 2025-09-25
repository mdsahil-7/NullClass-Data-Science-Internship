import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
from sign_language_detector import SignLanguageDetector
from datetime import datetime, time

class SignLanguageDetectionGUI:
    def __init__(self):
        # Initialize CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.window = ctk.CTk()
        self.window.title("🤟 Sign Language Detection System")
        self.window.geometry("1600x1000")

        self.detector = SignLanguageDetector()
        self.current_image_path = None
        self.latest_detections = []

        # Video capture variables
        self.cap = None
        self.is_video_running = False
        self.video_thread = None

        # Time status update
        self.time_update_job = None

        self.setup_gui()
        self.start_time_updates()

    def setup_gui(self):
        """Setup the GUI layout"""
        # Main title
        title_label = ctk.CTkLabel(
            self.window, 
            text="🤟 Sign Language Detection System",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(pady=20)

        subtitle_label = ctk.CTkLabel(
            self.window,
            text="AI-Powered ASL Recognition with Time-Based Operation (6:00 PM - 10:00 PM)",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        subtitle_label.pack(pady=(0, 20))

        # Operation status banner
        self.status_frame = ctk.CTkFrame(self.window, height=60)
        self.status_frame.pack(fill="x", padx=20, pady=(0, 10))
        self.status_frame.pack_propagate(False)

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="🟢 SYSTEM ACTIVE",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="green"
        )
        self.status_label.pack(side="left", padx=20, pady=15)

        self.time_label = ctk.CTkLabel(
            self.status_frame,
            text="Current Time: --:--",
            font=ctk.CTkFont(size=14)
        )
        self.time_label.pack(side="right", padx=20, pady=15)

        # Main container
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Left panel - Controls
        left_panel = ctk.CTkFrame(main_frame, width=400)
        left_panel.pack(side="left", fill="y", padx=10, pady=10)
        left_panel.pack_propagate(False)

        # Input selection section
        input_frame = ctk.CTkFrame(left_panel)
        input_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(input_frame, text="📁 Input Selection", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.upload_image_btn = ctk.CTkButton(
            input_frame, text="🖼️ Upload Image", 
            command=self.upload_image, height=40
        )
        self.upload_image_btn.pack(pady=5, padx=10, fill="x")

        self.start_camera_btn = ctk.CTkButton(
            input_frame, text="📹 Start Real-Time Camera", 
            command=self.toggle_camera, height=40,
            fg_color="orange", hover_color="darkorange"
        )
        self.start_camera_btn.pack(pady=5, padx=10, fill="x")

        # Detection controls
        detection_frame = ctk.CTkFrame(left_panel)
        detection_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(detection_frame, text="🎯 Detection Controls", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.detect_image_btn = ctk.CTkButton(
            detection_frame, text="🔍 Detect Signs in Image", 
            command=self.detect_signs_image, height=40, state="disabled"
        )
        self.detect_image_btn.pack(pady=5, padx=10, fill="x")

        # Audio settings
        audio_frame = ctk.CTkFrame(left_panel)
        audio_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(audio_frame, text="🔊 Audio Settings", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.tts_var = tk.BooleanVar(value=True)
        tts_check = ctk.CTkCheckBox(
            audio_frame, text="🎤 Text-to-Speech Output", 
            variable=self.tts_var
        )
        tts_check.pack(pady=5)

        # Sign vocabulary display
        vocab_frame = ctk.CTkFrame(left_panel)
        vocab_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(vocab_frame, text="🤟 Available Signs", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        vocab_text = ctk.CTkTextbox(vocab_frame, height=120)
        vocab_text.pack(fill="x", padx=10, pady=5)

        # Display available signs
        signs = self.detector.get_available_signs()
        vocab_content = "Supported ASL Signs:\n"
        vocab_content += "\n".join([f"• {sign}" for sign in signs[:10]])
        vocab_content += f"\n... and {len(signs) - 10} more"

        vocab_text.insert("1.0", vocab_content)
        vocab_text.configure(state="disabled")

        # Results panel
        results_frame = ctk.CTkFrame(left_panel)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(results_frame, text="📊 Detection Results", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Results text area
        self.results_text = ctk.CTkTextbox(results_frame, height=180)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Statistics display
        stats_frame = ctk.CTkFrame(results_frame)
        stats_frame.pack(fill="x", padx=10, pady=5)

        self.hands_label = ctk.CTkLabel(
            stats_frame, text="🤲 Hands Detected: 0", 
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.hands_label.pack(pady=2)

        self.signs_label = ctk.CTkLabel(
            stats_frame, text="🤟 Signs Recognized: 0", 
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.signs_label.pack(pady=2)

        self.confidence_label = ctk.CTkLabel(
            stats_frame, text="📊 Avg Confidence: 0%", 
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.confidence_label.pack(pady=2)

        # Right panel - Preview and Display
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(right_panel, text="🖥️ Live Preview & Detection", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Display frame
        self.display_frame = ctk.CTkFrame(right_panel, fg_color="gray20")
        self.display_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_label = ctk.CTkLabel(
            self.display_frame, 
            text="No input selected\n\n📸 Upload an image or start camera\n🤟 System recognizes 20+ ASL signs\n⏰ Active: 6:00 PM - 10:00 PM", 
            font=ctk.CTkFont(size=16)
        )
        self.image_label.pack(expand=True)

        # Control buttons
        control_frame = ctk.CTkFrame(right_panel)
        control_frame.pack(fill="x", padx=10, pady=5)

        self.save_results_btn = ctk.CTkButton(
            control_frame, text="💾 Save Results", 
            command=self.save_results, state="disabled"
        )
        self.save_results_btn.pack(side="left", padx=5)

        self.save_image_btn = ctk.CTkButton(
            control_frame, text="🖼️ Save Screenshot", 
            command=self.save_screenshot, state="disabled"
        )
        self.save_image_btn.pack(side="right", padx=5)

    def start_time_updates(self):
        """Start periodic time status updates"""
        self.update_time_status()

    def update_time_status(self):
        """Update time status and operation availability"""
        status = self.detector.get_operation_status()

        # Update time display
        self.time_label.configure(text=f"Current Time: {status['current_time']}")

        # Update status display
        if status['is_active']:
            self.status_label.configure(
                text="🟢 SYSTEM ACTIVE", 
                text_color="green"
            )
            self.status_frame.configure(fg_color="darkgreen")
        else:
            self.status_label.configure(
                text="🔴 SYSTEM INACTIVE", 
                text_color="red"
            )
            self.status_frame.configure(fg_color="darkred")

        # Schedule next update
        self.window.after(1000, self.update_time_status)  # Update every second

    def upload_image(self):
        """Upload image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image for Sign Language Detection",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = file_path
            self.load_image_preview(file_path)
            self.detect_image_btn.configure(state="normal")
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", f"📄 Image uploaded: {os.path.basename(file_path)}\n")

            # Check operation time
            status = self.detector.get_operation_status()
            if status['is_active']:
                self.results_text.insert("end", "✅ System is active. Ready for detection.\n")
            else:
                self.results_text.insert("end", f"⏰ System inactive. Active hours: {status['operation_start']} - {status['operation_end']}\n")

    def load_image_preview(self, image_path):
        """Load and display image preview"""
        try:
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                raise Exception("Could not load image")

            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Resize for display
            height, width = cv_image.shape[:2]
            max_size = 800
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            cv_image = cv2.resize(cv_image, (new_width, new_height))

            # Convert to PIL and display
            pil_image = Image.fromarray(cv_image)
            photo = ImageTk.PhotoImage(pil_image)

            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo

        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")

    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.is_video_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Start camera for real-time detection"""
        if not self.detector.is_operation_time():
            status = self.detector.get_operation_status()
            messagebox.showwarning(
                "System Inactive", 
                f"Camera can only be used during operation hours:\n{status['operation_start']} - {status['operation_end']}\n\nCurrent time: {status['current_time']}"
            )
            return

        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not access camera")
                return

            self.is_video_running = True
            self.start_camera_btn.configure(
                text="⏹️ Stop Camera", 
                fg_color="red", 
                hover_color="darkred"
            )

            # Start video processing thread
            self.video_thread = threading.Thread(target=self.video_loop)
            self.video_thread.daemon = True
            self.video_thread.start()

            self.results_text.insert("end", "📹 Camera started - Real-time detection active\n")

        except Exception as e:
            messagebox.showerror("Error", f"Could not start camera: {e}")

    def stop_camera(self):
        """Stop camera"""
        self.is_video_running = False

        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_camera_btn.configure(
            text="📹 Start Real-Time Camera", 
            fg_color="orange", 
            hover_color="darkorange"
        )

        self.results_text.insert("end", "⏹️ Camera stopped\n")

    def video_loop(self):
        """Main video processing loop"""
        while self.is_video_running and self.cap:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Process frame for sign detection
                annotated_frame, detections, status_message = self.detector.process_frame_for_signs(frame)

                if "not operational" in status_message:
                    # System became inactive during operation
                    self.window.after(0, self.stop_camera)
                    self.window.after(0, lambda: messagebox.showwarning("System Inactive", status_message))
                    break

                # Update display
                self.update_video_display(annotated_frame, detections)

                # Handle text-to-speech for detected signs
                if detections and self.tts_var.get():
                    for detection in detections:
                        if detection['confidence'] > 0.8:  # High confidence threshold for TTS
                            self.detector.speak_text(detection['sign'])

            except Exception as e:
                print(f"Video loop error: {e}")
                break

        # Cleanup
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_video_display(self, frame, detections):
        """Update video display with detection results"""
        try:
            # Convert frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize for display
            height, width = frame_rgb.shape[:2]
            max_size = 800
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))

            # Convert to PIL and display
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)

            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo

            # Update statistics
            if detections:
                self.latest_detections = detections
                summary = self.detector.get_detection_summary(detections)

                self.hands_label.configure(text=f"🤲 Hands Detected: {summary['total_hands']}")
                self.signs_label.configure(text=f"🤟 Signs Recognized: {len(summary['signs_detected'])}")
                self.confidence_label.configure(text=f"📊 Avg Confidence: {summary['average_confidence']*100:.0f}%")

                # Enable save buttons
                self.save_results_btn.configure(state="normal")
                self.save_image_btn.configure(state="normal")

        except Exception as e:
            print(f"Display update error: {e}")

    def detect_signs_image(self):
        """Detect signs in uploaded image"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please upload an image first")
            return

        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", "🔍 Detecting sign language...\n")
        self.window.update()

        def detection_thread():
            try:
                annotated_image, detections, status_message = self.detector.process_image_for_signs(self.current_image_path)

                if "not operational" in status_message:
                    self.results_text.insert("end", f"⏰ {status_message}\n")
                    return

                if annotated_image is not None:
                    self.latest_detections = detections
                    self.display_detection_results(annotated_image, detections)

                    # Text-to-speech for detected signs
                    if detections and self.tts_var.get():
                        signs_to_speak = [d['sign'] for d in detections if d['confidence'] > 0.7]
                        if signs_to_speak:
                            self.detector.speak_text(f"Detected signs: {', '.join(signs_to_speak)}")
                else:
                    self.results_text.insert("end", "❌ No hands or signs detected\n")

            except Exception as e:
                self.results_text.insert("end", f"❌ Detection error: {e}\n")

        thread = threading.Thread(target=detection_thread)
        thread.daemon = True
        thread.start()

    def display_detection_results(self, annotated_image, detections):
        """Display detection results in GUI"""
        # Update preview with annotated image
        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Resize for display
        height, width = annotated_rgb.shape[:2]
        max_size = 800
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        annotated_rgb = cv2.resize(annotated_rgb, (new_width, new_height))

        pil_image = Image.fromarray(annotated_rgb)
        photo = ImageTk.PhotoImage(pil_image)

        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo

        # Update results text
        summary = self.detector.get_detection_summary(detections)

        results_text = f"""
🤟 Sign Language Detection Complete!

📊 Summary:
• Hands Detected: {summary['total_hands']}
• Signs Recognized: {len(summary['signs_detected'])}
• Average Confidence: {summary['average_confidence']*100:.1f}%

🤲 Hand Analysis:
"""

        for hand_type, count in summary['hand_types'].items():
            results_text += f"• {hand_type} Hand: {count}\n"

        results_text += "\n🤟 Signs Detected:\n"
        for sign, count in summary['signs_detected'].items():
            results_text += f"• {sign}: {count} time{'s' if count > 1 else ''}\n"

        results_text += "\n👥 Individual Results:\n"
        for i, detection in enumerate(detections, 1):
            hand_type = detection['hand_type']
            sign = detection['sign']
            confidence = detection['confidence']

            results_text += f"• {hand_type} Hand {i}: {sign} ({confidence:.2f} confidence)\n"

        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", results_text)

        # Update statistics labels
        self.hands_label.configure(text=f"🤲 Hands Detected: {summary['total_hands']}")
        self.signs_label.configure(text=f"🤟 Signs Recognized: {len(summary['signs_detected'])}")
        self.confidence_label.configure(text=f"📊 Avg Confidence: {summary['average_confidence']*100:.0f}%")

        # Enable save buttons
        self.save_results_btn.configure(state="normal")
        self.save_image_btn.configure(state="normal")

    def save_results(self):
        """Save detection results"""
        if not self.latest_detections:
            messagebox.showwarning("Warning", "No results to save")
            return

        try:
            if self.current_image_path:
                saved_path = self.detector.save_results(self.current_image_path, self.latest_detections)
            else:
                # For camera captures, create temporary image path
                temp_path = f"camera_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                saved_path = self.detector.save_results(temp_path, self.latest_detections)

            messagebox.showinfo("Success", 
                               f"Sign language detection results saved to:\n{saved_path}\n\n" +
                               "Contains hand landmarks, signs detected, and confidence scores")

        except Exception as e:
            messagebox.showerror("Error", f"Could not save results: {e}")

    def save_screenshot(self):
        """Save current screenshot"""
        if not hasattr(self.image_label, 'image') or not self.image_label.image:
            messagebox.showwarning("Warning", "No image to save")
            return

        try:
            save_path = filedialog.asksaveasfilename(
                title="Save Screenshot",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
            )

            if save_path:
                # Get current image from label
                pil_image = ImageTk.getimage(self.image_label.image)
                pil_image.save(save_path)
                messagebox.showinfo("Success", f"Screenshot saved to:\n{save_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not save screenshot: {e}")

    def on_closing(self):
        """Handle window closing"""
        if self.is_video_running:
            self.stop_camera()
        self.window.destroy()

    def run(self):
        """Start the GUI application"""
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

if __name__ == "__main__":
    app = SignLanguageDetectionGUI()
    app.run()
