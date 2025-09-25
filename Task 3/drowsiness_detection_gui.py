import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
from drowsiness_detector import DrowsinessDetector
import pygame
from datetime import datetime

class DrowsinessDetectionGUI:
    def __init__(self):
        # Initialize CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.window = ctk.CTk()
        self.window.title("😴 Drowsiness Detection System - Vehicle Safety")
        self.window.geometry("1500x900")

        # Initialize pygame for sound alerts
        try:
            pygame.mixer.init()
            self.sound_enabled = True
        except:
            self.sound_enabled = False
            print("⚠️ Sound system not available")

        self.detector = DrowsinessDetector()
        self.current_image_path = None
        self.current_video_path = None
        self.is_video_playing = False
        self.video_thread = None
        self.latest_detections = []

        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI layout"""
        # Main title
        title_label = ctk.CTkLabel(
            self.window, 
            text="😴 Drowsiness Detection System - Vehicle Safety Monitor",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=20)

        # Main container
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Left panel - Controls
        left_panel = ctk.CTkFrame(main_frame, width=380)
        left_panel.pack(side="left", fill="y", padx=10, pady=10)
        left_panel.pack_propagate(False)

        # File selection section
        file_frame = ctk.CTkFrame(left_panel)
        file_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(file_frame, text="📁 Media Selection", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.select_image_btn = ctk.CTkButton(
            file_frame, text="🖼️ Select Image", 
            command=self.select_image, height=40
        )
        self.select_image_btn.pack(pady=5, padx=10, fill="x")

        self.select_video_btn = ctk.CTkButton(
            file_frame, text="🎥 Select Video", 
            command=self.select_video, height=40
        )
        self.select_video_btn.pack(pady=5, padx=10, fill="x")

        # Detection settings
        settings_frame = ctk.CTkFrame(left_panel)
        settings_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(settings_frame, text="⚙️ Detection Settings", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Eye closure threshold
        ctk.CTkLabel(settings_frame, text="Eye Closure Threshold:").pack(pady=(5, 0))
        self.eye_threshold_var = tk.DoubleVar(value=0.25)
        self.eye_threshold_slider = ctk.CTkSlider(
            settings_frame, from_=0.15, to=0.35, 
            variable=self.eye_threshold_var, number_of_steps=10
        )
        self.eye_threshold_slider.pack(pady=5, padx=10, fill="x")

        self.threshold_label = ctk.CTkLabel(settings_frame, text="0.25")
        self.threshold_label.pack()

        # Update threshold display
        def update_threshold(*args):
            self.threshold_label.configure(text=f"{self.eye_threshold_var.get():.2f}")
            if hasattr(self, 'detector'):
                self.detector.EYE_AR_THRESH = self.eye_threshold_var.get()
        self.eye_threshold_var.trace_add("write", update_threshold)

        # Sound alerts toggle
        self.sound_var = tk.BooleanVar(value=True)
        sound_check = ctk.CTkCheckBox(
            settings_frame, text="🔊 Sound Alerts", 
            variable=self.sound_var
        )
        sound_check.pack(pady=5)

        # Detection buttons
        detect_frame = ctk.CTkFrame(left_panel)
        detect_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(detect_frame, text="🎯 Detection Controls", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.detect_image_btn = ctk.CTkButton(
            detect_frame, text="🔍 Analyze Image", 
            command=self.detect_drowsiness_image, height=40, state="disabled"
        )
        self.detect_image_btn.pack(pady=5, padx=10, fill="x")

        self.detect_video_btn = ctk.CTkButton(
            detect_frame, text="🎬 Process Video", 
            command=self.process_video, height=40, state="disabled"
        )
        self.detect_video_btn.pack(pady=5, padx=10, fill="x")

        # Results panel
        results_frame = ctk.CTkFrame(left_panel)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(results_frame, text="📊 Detection Results", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Results text area
        self.results_text = ctk.CTkTextbox(results_frame, height=200)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Alert panels
        alert_frame = ctk.CTkFrame(results_frame)
        alert_frame.pack(fill="x", padx=10, pady=5)

        # People count
        self.people_label = ctk.CTkLabel(
            alert_frame, text="👥 People: 0", 
            font=ctk.CTkFont(size=14, weight="bold"), text_color="blue"
        )
        self.people_label.pack(pady=5)

        # Sleeping count
        self.sleeping_label = ctk.CTkLabel(
            alert_frame, text="😴 Sleeping: 0", 
            font=ctk.CTkFont(size=14, weight="bold"), text_color="red"
        )
        self.sleeping_label.pack(pady=5)

        # Right panel - Preview
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(right_panel, text="🖥️ Live Preview", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Image/Video display
        self.display_frame = ctk.CTkFrame(right_panel, fg_color="gray20")
        self.display_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_label = ctk.CTkLabel(
            self.display_frame, text="No media selected\nSupports: JPG, PNG, MP4, AVI, MOV", 
            font=ctk.CTkFont(size=16)
        )
        self.image_label.pack(expand=True)

        # Control panel
        control_frame = ctk.CTkFrame(right_panel)
        control_frame.pack(fill="x", padx=10, pady=5)

        # Video controls
        video_controls = ctk.CTkFrame(control_frame)
        video_controls.pack(side="left", fill="x", expand=True, padx=5)

        self.play_pause_btn = ctk.CTkButton(
            video_controls, text="▶️ Play", 
            command=self.toggle_video, state="disabled"
        )
        self.play_pause_btn.pack(side="left", padx=5)

        self.stop_btn = ctk.CTkButton(
            video_controls, text="⏹️ Stop", 
            command=self.stop_video, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)

        # Export controls
        export_controls = ctk.CTkFrame(control_frame)
        export_controls.pack(side="right", padx=5)

        self.save_results_btn = ctk.CTkButton(
            export_controls, text="💾 Save Results", 
            command=self.save_results, state="disabled"
        )
        self.save_results_btn.pack(side="right", padx=5)

    def select_image(self):
        """Select image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = file_path
            self.current_video_path = None
            self.load_image_preview(file_path)
            self.detect_image_btn.configure(state="normal")
            self.play_pause_btn.configure(state="disabled")
            self.stop_btn.configure(state="disabled")
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", f"📄 Image selected: {os.path.basename(file_path)}\n")

    def select_video(self):
        """Select video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_video_path = file_path
            self.current_image_path = None
            self.load_video_preview(file_path)
            self.detect_video_btn.configure(state="normal")
            self.play_pause_btn.configure(state="normal")
            self.stop_btn.configure(state="normal")
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", f"🎥 Video selected: {os.path.basename(file_path)}\n")

    def load_image_preview(self, image_path):
        """Load and display image preview"""
        try:
            # Load image with OpenCV
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                raise Exception("Could not load image")

            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Resize for display
            height, width = cv_image.shape[:2]
            max_size = 700
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
            self.image_label.image = photo  # Keep reference

        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")

    def load_video_preview(self, video_path):
        """Load first frame of video as preview"""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize for display
                height, width = frame.shape[:2]
                max_size = 700
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))

                frame = cv2.resize(frame, (new_width, new_height))

                # Convert to PIL and display
                pil_image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(pil_image)

                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo

                # Add video info
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0

                info_text = f"Video: {duration:.1f}s, {int(fps)} FPS"
                self.results_text.insert("end", f"ℹ️ {info_text}\n")

            cap.release()

        except Exception as e:
            messagebox.showerror("Error", f"Could not load video: {e}")

    def detect_drowsiness_image(self):
        """Run drowsiness detection on selected image"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return

        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", "😴 Analyzing image for drowsiness...\n")
        self.window.update()

        def detection_thread():
            try:
                # Update detector settings
                self.detector.EYE_AR_THRESH = self.eye_threshold_var.get()

                # Run detection
                annotated_image, detections, total_people, sleeping_count = self.detector.detect_drowsiness_in_image(self.current_image_path)

                if annotated_image is not None:
                    self.latest_detections = detections

                    # Display results
                    self.display_detection_results(annotated_image, detections, total_people, sleeping_count)

                    # Show alerts if people are sleeping
                    if sleeping_count > 0:
                        sleeping_people = [d for d in detections if d['is_sleeping']]
                        self.show_drowsiness_alert(sleeping_count, sleeping_people)

                else:
                    self.results_text.insert("end", "❌ Detection failed\n")

            except Exception as e:
                self.results_text.insert("end", f"❌ Error: {e}\n")

        thread = threading.Thread(target=detection_thread)
        thread.daemon = True
        thread.start()

    def process_video(self):
        """Process video for drowsiness detection"""
        if not self.current_video_path:
            messagebox.showwarning("Warning", "Please select a video first")
            return

        output_path = filedialog.asksaveasfilename(
            title="Save Processed Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")]
        )

        if output_path:
            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", "🎬 Processing video for drowsiness detection...\n")

            def video_thread():
                try:
                    self.detector.EYE_AR_THRESH = self.eye_threshold_var.get()
                    max_sleeping = 0
                    max_sleeping_ages = []

                    # Process video frames
                    for result in self.detector.process_video_for_drowsiness(self.current_video_path, output_path):
                        annotated_frame, detections, total_people, sleeping_count, frame_num, total_frames = result

                        # Update display periodically
                        if frame_num % 30 == 0:  # Every 30 frames
                            # Show frame
                            self.display_frame_result(annotated_frame, detections, total_people, sleeping_count)

                            # Update progress
                            progress = (frame_num / total_frames) * 100
                            self.results_text.delete("end-2l", "end-1l")
                            self.results_text.insert("end", f"Progress: {progress:.1f}% | Frame {frame_num}/{total_frames}\n")

                        # Track maximum sleeping people
                        if sleeping_count > max_sleeping:
                            max_sleeping = sleeping_count
                            max_sleeping_ages = [d['age'] for d in detections if d['is_sleeping']]

                    self.results_text.insert("end", f"✅ Video processing completed!\n")
                    self.results_text.insert("end", f"💾 Saved: {output_path}\n")
                    self.results_text.insert("end", f"📊 Max sleeping people: {max_sleeping}\n")

                    if max_sleeping > 0:
                        self.show_video_drowsiness_alert(max_sleeping, max_sleeping_ages)

                except Exception as e:
                    self.results_text.insert("end", f"❌ Video processing error: {e}\n")

            self.video_thread = threading.Thread(target=video_thread)
            self.video_thread.daemon = True
            self.video_thread.start()

    def display_detection_results(self, annotated_image, detections, total_people, sleeping_count):
        """Display detection results in GUI"""
        # Update preview with annotated image
        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Resize for display
        height, width = annotated_rgb.shape[:2]
        max_size = 700
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
😴 Drowsiness Detection Complete!

📊 Summary:
• Total People: {summary['total_people']}
• 😴 Sleeping: {summary['sleeping_count']} people
• 😊 Awake: {summary['awake_count']} people
• 📈 Average Age: {summary['average_age']:.1f} years

👥 Detailed Results:
"""

        for i, detection in enumerate(detections, 1):
            status = "😴 SLEEPING" if detection['is_sleeping'] else "😊 AWAKE"
            color_indicator = "🔴" if detection['is_sleeping'] else "🟢"
            results_text += f"  {color_indicator} Person {i}: {status}, Age: {detection['age']}\n"

        if sleeping_count > 0:
            results_text += f"\n⚠️ WARNING: {sleeping_count} person(s) detected sleeping!\n"
            sleeping_ages = [str(d['age']) for d in detections if d['is_sleeping']]
            results_text += f"Ages of sleeping people: {', '.join(sleeping_ages)}\n"

        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", results_text)

        # Update counters
        self.people_label.configure(text=f"👥 People: {total_people}")
        self.sleeping_label.configure(text=f"😴 Sleeping: {sleeping_count}")

        # Enable save button
        self.save_results_btn.configure(state="normal")

    def display_frame_result(self, annotated_frame, detections, total_people, sleeping_count):
        """Display single frame result during video processing"""
        # Update preview
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        height, width = annotated_rgb.shape[:2]
        max_size = 700
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

        # Update counters
        self.people_label.configure(text=f"👥 People: {total_people}")
        self.sleeping_label.configure(text=f"😴 Sleeping: {sleeping_count}")

        self.window.update()

    def show_drowsiness_alert(self, sleeping_count, sleeping_people):
        """Show popup alert for sleeping people with ages"""
        # Play sound alert
        if self.sound_var.get() and self.sound_enabled:
            try:
                # Create a simple beep sound
                frequency = 800  # Hz
                duration = 500  # milliseconds
                # Note: This requires additional sound implementation
                pass
            except:
                pass

        popup = ctk.CTkToplevel(self.window)
        popup.title("🚨 Drowsiness Alert!")
        popup.geometry("500x400")
        popup.transient(self.window)
        popup.grab_set()

        # Center the popup
        popup.geometry("+%d+%d" % (
            self.window.winfo_rootx() + 400,
            self.window.winfo_rooty() + 200
        ))

        alert_frame = ctk.CTkFrame(popup, fg_color="darkred")
        alert_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Alert title
        ctk.CTkLabel(
            alert_frame, 
            text="🚨 DROWSINESS DETECTED! 🚨",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="white"
        ).pack(pady=20)

        # Alert message
        ctk.CTkLabel(
            alert_frame,
            text=f"{sleeping_count} person{'s' if sleeping_count > 1 else ''} detected sleeping in vehicle!",
            font=ctk.CTkFont(size=16),
            text_color="white"
        ).pack(pady=10)

        # Age information
        age_info = ctk.CTkTextbox(alert_frame, height=150, width=400)
        age_info.pack(pady=10, padx=20, fill="both", expand=True)

        age_text = "👥 Sleeping People Details:\n\n"
        for i, person in enumerate(sleeping_people, 1):
            age_text += f"Person {person['person_id']}: Age {person['age']} years\n"
            age_text += f"  Confidence: {person['confidence']:.2f}\n"
            age_text += f"  Detection: {person['method']}\n\n"

        age_info.insert("1.0", age_text)
        age_info.configure(state="disabled")

        # Buttons
        button_frame = ctk.CTkFrame(alert_frame)
        button_frame.pack(pady=20)

        ctk.CTkButton(
            button_frame,
            text="🚨 WAKE DRIVER",
            command=popup.destroy,
            fg_color="yellow",
            text_color="darkred",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            button_frame,
            text="📞 EMERGENCY CALL",
            command=lambda: [popup.destroy(), self.show_emergency_dialog()],
            fg_color="orange",
            text_color="darkred",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            button_frame,
            text="✅ ACKNOWLEDGE",
            command=popup.destroy,
            fg_color="white",
            text_color="darkred"
        ).pack(side="left", padx=10)

    def show_video_drowsiness_alert(self, max_sleeping, ages):
        """Show alert for video processing results"""
        popup = ctk.CTkToplevel(self.window)
        popup.title("📊 Video Analysis Complete")
        popup.geometry("450x300")
        popup.transient(self.window)
        popup.grab_set()

        popup.geometry("+%d+%d" % (
            self.window.winfo_rootx() + 400,
            self.window.winfo_rooty() + 250
        ))

        alert_frame = ctk.CTkFrame(popup, fg_color="navy")
        alert_frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            alert_frame, 
            text="📊 Video Analysis Complete",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="white"
        ).pack(pady=20)

        ctk.CTkLabel(
            alert_frame,
            text=f"Maximum {max_sleeping} person{'s' if max_sleeping > 1 else ''} sleeping simultaneously",
            font=ctk.CTkFont(size=14),
            text_color="white"
        ).pack(pady=10)

        if ages:
            ages_text = f"Ages: {', '.join(map(str, ages))} years"
            ctk.CTkLabel(
                alert_frame,
                text=ages_text,
                font=ctk.CTkFont(size=12),
                text_color="white"
            ).pack(pady=5)

        ctk.CTkButton(
            alert_frame,
            text="OK",
            command=popup.destroy,
            fg_color="white",
            text_color="navy"
        ).pack(pady=20)

    def show_emergency_dialog(self):
        """Show emergency contact dialog"""
        messagebox.showinfo("Emergency", 
                           "🚨 EMERGENCY PROTOCOL ACTIVATED\n\n" +
                           "Recommended Actions:\n" +
                           "• Wake the driver immediately\n" +
                           "• Pull over safely if possible\n" +
                           "• Call emergency services: 911\n" +
                           "• Contact vehicle fleet manager\n\n" +
                           "This is a demonstration system.")

    def toggle_video(self):
        """Toggle video play/pause"""
        if self.current_video_path:
            if not self.is_video_playing:
                self.play_video()
            else:
                self.is_video_playing = False
                self.play_pause_btn.configure(text="▶️ Play")

    def play_video(self):
        """Play video in preview mode"""
        self.is_video_playing = True
        self.play_pause_btn.configure(text="⏸️ Pause")

        def video_player():
            cap = cv2.VideoCapture(self.current_video_path)

            while self.is_video_playing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize for display
                height, width = frame.shape[:2]
                max_size = 700
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))

                frame = cv2.resize(frame, (new_width, new_height))

                pil_image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(pil_image)

                self.image_label.configure(image=photo)
                self.image_label.image = photo

                self.window.update()
                self.window.after(33)  # ~30 FPS

            cap.release()
            if self.is_video_playing:  # Video ended naturally
                self.is_video_playing = False
                self.play_pause_btn.configure(text="▶️ Play")

        video_thread = threading.Thread(target=video_player)
        video_thread.daemon = True
        video_thread.start()

    def stop_video(self):
        """Stop video playback"""
        self.is_video_playing = False
        self.play_pause_btn.configure(text="▶️ Play")
        if self.current_video_path:
            self.load_video_preview(self.current_video_path)

    def save_results(self):
        """Save detection results"""
        if not self.latest_detections:
            messagebox.showwarning("Warning", "No results to save")
            return

        try:
            if self.current_image_path:
                saved_path = self.detector.save_results(self.current_image_path, self.latest_detections)

                # Save annotated image
                annotated_image, _, _, _ = self.detector.detect_drowsiness_in_image(self.current_image_path)
                if annotated_image is not None:
                    output_image_path = saved_path.replace('.json', '_annotated.jpg')
                    cv2.imwrite(output_image_path, annotated_image)

                messagebox.showinfo("Success", 
                                   f"Results saved to:\n{saved_path}\n\n" +
                                   f"Annotated image: {output_image_path if annotated_image is not None else 'Not saved'}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not save results: {e}")

    def run(self):
        """Start the GUI application"""
        self.window.mainloop()

if __name__ == "__main__":
    app = DrowsinessDetectionGUI()
    app.run()
