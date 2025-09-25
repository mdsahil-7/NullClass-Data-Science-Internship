import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
from animal_detector import AnimalDetector

class AnimalDetectionGUI:
    def __init__(self):
        # Initialize CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.window = ctk.CTk()
        self.window.title("🦁 Animal Detection System")
        self.window.geometry("1400x900")

        self.detector = AnimalDetector()
        self.current_image = None
        self.current_video_path = None
        self.is_video_playing = False

        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI layout"""
        # Main title
        title_label = ctk.CTkLabel(
            self.window, 
            text="🦁 Animal Detection & Classification System",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=20)

        # Main container
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Left panel - Controls
        left_panel = ctk.CTkFrame(main_frame, width=350)
        left_panel.pack(side="left", fill="y", padx=10, pady=10)
        left_panel.pack_propagate(False)

        # Control buttons
        controls_frame = ctk.CTkFrame(left_panel)
        controls_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(controls_frame, text="📁 File Selection", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.select_image_btn = ctk.CTkButton(
            controls_frame, text="🖼️ Select Image", 
            command=self.select_image, height=40
        )
        self.select_image_btn.pack(pady=5, padx=10, fill="x")

        self.select_video_btn = ctk.CTkButton(
            controls_frame, text="🎥 Select Video", 
            command=self.select_video, height=40
        )
        self.select_video_btn.pack(pady=5, padx=10, fill="x")

        # Detection controls
        detection_frame = ctk.CTkFrame(left_panel)
        detection_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(detection_frame, text="🎯 Detection Controls", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Confidence threshold
        ctk.CTkLabel(detection_frame, text="Confidence Threshold:").pack(pady=(5, 0))
        self.confidence_var = tk.DoubleVar(value=0.5)
        self.confidence_slider = ctk.CTkSlider(
            detection_frame, from_=0.1, to=1.0, 
            variable=self.confidence_var, number_of_steps=9
        )
        self.confidence_slider.pack(pady=5, padx=10, fill="x")

        self.confidence_label = ctk.CTkLabel(detection_frame, text="0.5")
        self.confidence_label.pack()

        # Update confidence display
        def update_confidence(*args):
            self.confidence_label.configure(text=f"{self.confidence_var.get():.1f}")
        self.confidence_var.trace_add("write", update_confidence)

        # Detect buttons
        self.detect_image_btn = ctk.CTkButton(
            detection_frame, text="🔍 Detect Animals in Image", 
            command=self.detect_image, height=40, state="disabled"
        )
        self.detect_image_btn.pack(pady=5, padx=10, fill="x")

        self.detect_video_btn = ctk.CTkButton(
            detection_frame, text="🎬 Process Video", 
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

        # Carnivore alert
        self.carnivore_alert = ctk.CTkFrame(results_frame)
        self.carnivore_alert.pack(fill="x", padx=10, pady=5)

        self.carnivore_label = ctk.CTkLabel(
            self.carnivore_alert, text="🔴 Carnivores: 0", 
            font=ctk.CTkFont(size=14, weight="bold"), text_color="red"
        )
        self.carnivore_label.pack(pady=10)

        # Right panel - Preview
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(right_panel, text="🖥️ Preview", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Image/Video display
        self.display_frame = ctk.CTkFrame(right_panel, fg_color="gray20")
        self.display_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_label = ctk.CTkLabel(
            self.display_frame, text="No image/video selected", 
            font=ctk.CTkFont(size=16)
        )
        self.image_label.pack(expand=True)

        # Video controls
        self.video_controls = ctk.CTkFrame(right_panel)
        self.video_controls.pack(fill="x", padx=10, pady=5)

        self.play_pause_btn = ctk.CTkButton(
            self.video_controls, text="▶️ Play", 
            command=self.toggle_video, state="disabled"
        )
        self.play_pause_btn.pack(side="left", padx=5)

        self.save_results_btn = ctk.CTkButton(
            self.video_controls, text="💾 Save Results", 
            command=self.save_results, state="disabled"
        )
        self.save_results_btn.pack(side="right", padx=5)

    def select_image(self):
        """Select image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )

        if file_path:
            self.current_image_path = file_path
            self.load_image_preview(file_path)
            self.detect_image_btn.configure(state="normal")
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", f"Image selected: {os.path.basename(file_path)}\n")

    def select_video(self):
        """Select video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")]
        )

        if file_path:
            self.current_video_path = file_path
            self.load_video_preview(file_path)
            self.detect_video_btn.configure(state="normal")
            self.play_pause_btn.configure(state="normal")
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", f"Video selected: {os.path.basename(file_path)}\n")

    def load_image_preview(self, image_path):
        """Load and display image preview"""
        try:
            # Load image with OpenCV
            cv_image = cv2.imread(image_path)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Resize for display
            height, width = cv_image.shape[:2]
            max_size = 600
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
                max_size = 600
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

            cap.release()

        except Exception as e:
            messagebox.showerror("Error", f"Could not load video: {e}")

    def detect_image(self):
        """Run animal detection on selected image"""
        if not hasattr(self, 'current_image_path'):
            messagebox.showwarning("Warning", "Please select an image first")
            return

        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", "🔍 Detecting animals...\n")
        self.window.update()

        def detection_thread():
            try:
                confidence = self.confidence_var.get()
                annotated_image, detections, carnivore_count = self.detector.detect_animals_image(
                    self.current_image_path, confidence
                )

                if annotated_image is not None:
                    # Display results
                    self.display_detection_results(annotated_image, detections, carnivore_count)

                    # Show carnivore popup if any found
                    if carnivore_count > 0:
                        self.show_carnivore_popup(carnivore_count)

                else:
                    self.results_text.insert("end", "❌ Detection failed\n")

            except Exception as e:
                self.results_text.insert("end", f"❌ Error: {e}\n")

        thread = threading.Thread(target=detection_thread)
        thread.daemon = True
        thread.start()

    def process_video(self):
        """Process video for animal detection"""
        if not self.current_video_path:
            messagebox.showwarning("Warning", "Please select a video first")
            return

        output_path = filedialog.asksaveasfilename(
            title="Save Processed Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")]
        )

        if output_path:
            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", "🎬 Processing video...\n")

            def video_thread():
                try:
                    confidence = self.confidence_var.get()
                    result, max_carnivores = self.detector.detect_animals_video(
                        self.current_video_path, output_path, confidence
                    )

                    if result:
                        self.results_text.insert("end", f"✅ Video saved: {output_path}\n")
                        self.results_text.insert("end", f"🔴 Max carnivores detected: {max_carnivores}\n")

                        if max_carnivores > 0:
                            self.show_carnivore_popup(max_carnivores, is_video=True)
                    else:
                        self.results_text.insert("end", "❌ Video processing failed\n")

                except Exception as e:
                    self.results_text.insert("end", f"❌ Error: {e}\n")

            thread = threading.Thread(target=video_thread)
            thread.daemon = True
            thread.start()

    def display_detection_results(self, annotated_image, detections, carnivore_count):
        """Display detection results"""
        # Update preview with annotated image
        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Resize for display
        height, width = annotated_rgb.shape[:2]
        max_size = 600
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
🎯 Detection Complete!

📊 Summary:
• Total Animals: {summary['total_animals']}
• Carnivores: {summary['carnivores_count']} 🔴
• Herbivores: {summary['herbivores_count']} 🟢

🦁 Species Detected:
"""

        for species, count in summary['species_detected'].items():
            is_carn = species.lower() in [c.lower() for c in summary['carnivore_species']]
            emoji = "🔴" if is_carn else "🟢"
            results_text += f"• {species}: {count} {emoji}\n"

        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", results_text)

        # Update carnivore counter
        self.carnivore_label.configure(text=f"🔴 Carnivores: {carnivore_count}")
        self.save_results_btn.configure(state="normal")

    def show_carnivore_popup(self, count, is_video=False):
        """Show popup alert for carnivorous animals"""
        media_type = "video" if is_video else "image"

        popup = ctk.CTkToplevel(self.window)
        popup.title("⚠️ Carnivore Alert!")
        popup.geometry("400x200")
        popup.transient(self.window)
        popup.grab_set()

        # Center the popup
        popup.geometry("+%d+%d" % (
            self.window.winfo_rootx() + 500,
            self.window.winfo_rooty() + 300
        ))

        alert_frame = ctk.CTkFrame(popup, fg_color="darkred")
        alert_frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            alert_frame, 
            text="⚠️ CARNIVORE DETECTED! ⚠️",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="white"
        ).pack(pady=20)

        ctk.CTkLabel(
            alert_frame,
            text=f"{count} carnivorous animal{'s' if count > 1 else ''} detected in {media_type}!",
            font=ctk.CTkFont(size=14),
            text_color="white"
        ).pack(pady=10)

        ctk.CTkButton(
            alert_frame,
            text="OK",
            command=popup.destroy,
            fg_color="white",
            text_color="darkred"
        ).pack(pady=20)

    def toggle_video(self):
        """Toggle video play/pause"""
        if hasattr(self, 'current_video_path') and self.current_video_path:
            if not self.is_video_playing:
                self.play_video()
            else:
                self.is_video_playing = False
                self.play_pause_btn.configure(text="▶️ Play")

    def play_video(self):
        """Play video in preview"""
        self.is_video_playing = True
        self.play_pause_btn.configure(text="⏸️ Pause")

        def video_player():
            cap = cv2.VideoCapture(self.current_video_path)

            while self.is_video_playing:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (600, 400))

                pil_image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(pil_image)

                self.image_label.configure(image=photo)
                self.image_label.image = photo

                self.window.update()
                self.window.after(33)  # ~30 FPS

            cap.release()

        thread = threading.Thread(target=video_player)
        thread.daemon = True
        thread.start()

    def save_results(self):
        """Save detection results"""
        if hasattr(self, 'current_image_path'):
            try:
                # Re-run detection to get results for saving
                annotated_image, detections, carnivore_count = self.detector.detect_animals_image(
                    self.current_image_path, self.confidence_var.get()
                )

                if detections:
                    saved_path = self.detector.save_results(self.current_image_path, detections)

                    # Save annotated image
                    output_image_path = saved_path.replace('.json', '_annotated.jpg')
                    cv2.imwrite(output_image_path, annotated_image)

                    messagebox.showinfo("Success", f"Results saved to:\n{saved_path}\n{output_image_path}")

            except Exception as e:
                messagebox.showerror("Error", f"Could not save results: {e}")

    def run(self):
        """Start the GUI application"""
        self.window.mainloop()

if __name__ == "__main__":
    app = AnimalDetectionGUI()
    app.run()
