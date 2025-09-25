import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
from nationality_detector import NationalityDetector
from datetime import datetime

class NationalityDetectionGUI:
    def __init__(self):
        # Initialize CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.window = ctk.CTk()
        self.window.title("🌍 Nationality & Emotion Detection System")
        self.window.geometry("1600x1000")

        self.detector = NationalityDetector()
        self.current_image_path = None
        self.latest_detections = []

        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI layout"""
        # Main title
        title_label = ctk.CTkLabel(
            self.window, 
            text="🌍 Nationality & Emotion Detection System",
            font=ctk.CTkFont(size=26, weight="bold")
        )
        title_label.pack(pady=20)

        subtitle_label = ctk.CTkLabel(
            self.window,
            text="Advanced AI for Nationality, Emotion, Age & Dress Analysis",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        subtitle_label.pack(pady=(0, 20))

        # Main container
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Left panel - Controls
        left_panel = ctk.CTkFrame(main_frame, width=400)
        left_panel.pack(side="left", fill="y", padx=10, pady=10)
        left_panel.pack_propagate(False)

        # File selection section
        file_frame = ctk.CTkFrame(left_panel)
        file_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(file_frame, text="📁 Image Selection", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.select_image_btn = ctk.CTkButton(
            file_frame, text="🖼️ Upload Image", 
            command=self.select_image, height=40
        )
        self.select_image_btn.pack(pady=5, padx=10, fill="x")

        # Detection controls
        detection_frame = ctk.CTkFrame(left_panel)
        detection_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(detection_frame, text="🎯 Detection Controls", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.analyze_btn = ctk.CTkButton(
            detection_frame, text="🔍 Analyze Nationality & Emotion", 
            command=self.analyze_image, height=50, state="disabled",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.analyze_btn.pack(pady=10, padx=10, fill="x")

        # Nationality rules info
        rules_frame = ctk.CTkFrame(left_panel)
        rules_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(rules_frame, text="📋 Detection Rules", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        rules_text = ctk.CTkTextbox(rules_frame, height=120)
        rules_text.pack(fill="x", padx=10, pady=5)

        rules_content = """🇮🇳 Indian: Nationality + Emotion + Age + Dress Color
🇺🇸 American: Nationality + Emotion + Age  
🌍 African: Nationality + Emotion + Dress Color
🌎 Others: Nationality + Emotion Only

Color-coded bounding boxes by nationality"""

        rules_text.insert("1.0", rules_content)
        rules_text.configure(state="disabled")

        # Results panel
        results_frame = ctk.CTkFrame(left_panel)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(results_frame, text="📊 Detection Results", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Results text area
        self.results_text = ctk.CTkTextbox(results_frame, height=200)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Summary stats
        stats_frame = ctk.CTkFrame(results_frame)
        stats_frame.pack(fill="x", padx=10, pady=5)

        # People count
        self.people_label = ctk.CTkLabel(
            stats_frame, text="👥 People: 0", 
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.people_label.pack(pady=2)

        # Top nationality
        self.nationality_label = ctk.CTkLabel(
            stats_frame, text="🌍 Top Nationality: None", 
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.nationality_label.pack(pady=2)

        # Top emotion
        self.emotion_label = ctk.CTkLabel(
            stats_frame, text="😊 Top Emotion: None", 
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.emotion_label.pack(pady=2)

        # Right panel - Preview and Results
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(right_panel, text="🖥️ Image Preview & Analysis", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Image display
        self.display_frame = ctk.CTkFrame(right_panel, fg_color="gray20")
        self.display_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_label = ctk.CTkLabel(
            self.display_frame, 
            text="No image selected\n\nSupported formats: JPG, PNG, BMP\nDetects: Nationality, Emotion, Age, Dress Color", 
            font=ctk.CTkFont(size=16)
        )
        self.image_label.pack(expand=True)

        # Control buttons
        control_frame = ctk.CTkFrame(right_panel)
        control_frame.pack(fill="x", padx=10, pady=5)

        self.save_results_btn = ctk.CTkButton(
            control_frame, text="💾 Save Analysis Results", 
            command=self.save_results, state="disabled"
        )
        self.save_results_btn.pack(side="left", padx=5)

        self.save_image_btn = ctk.CTkButton(
            control_frame, text="🖼️ Save Annotated Image", 
            command=self.save_annotated_image, state="disabled"
        )
        self.save_image_btn.pack(side="right", padx=5)

    def select_image(self):
        """Select image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image for Nationality Detection",
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
            self.analyze_btn.configure(state="normal")
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", f"📄 Image selected: {os.path.basename(file_path)}\n")
            self.results_text.insert("end", "Click 'Analyze' to start nationality and emotion detection...\n")

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
            self.image_label.image = photo  # Keep reference

        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")

    def analyze_image(self):
        """Run nationality and emotion detection on selected image"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return

        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", "🔍 Analyzing nationality, emotion, age and dress...\n")
        self.results_text.insert("end", "Processing facial features...\n")
        self.window.update()

        def analysis_thread():
            try:
                # Run detection
                annotated_image, detections = self.detector.process_image(self.current_image_path)

                if annotated_image is not None:
                    self.latest_detections = detections

                    # Display results
                    self.display_analysis_results(annotated_image, detections)

                    # Show detailed popup for each detection
                    if detections:
                        self.show_detailed_results_popup(detections)

                else:
                    self.results_text.insert("end", "❌ Analysis failed - no faces detected\n")

            except Exception as e:
                self.results_text.insert("end", f"❌ Error during analysis: {e}\n")

        thread = threading.Thread(target=analysis_thread)
        thread.daemon = True
        thread.start()

    def display_analysis_results(self, annotated_image, detections):
        """Display analysis results in GUI"""
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
🎯 Analysis Complete!

📊 Summary:
• Total People: {summary['total_people']}
• Nationalities Detected: {len(summary['nationalities'])}
• Emotions Detected: {len(summary['emotions'])}

🌍 Nationality Breakdown:
"""

        for nationality, count in summary['nationalities'].items():
            results_text += f"  • {nationality}: {count} person{'s' if count > 1 else ''}\n"

        results_text += "\n😊 Emotion Breakdown:\n"
        for emotion, count in summary['emotions'].items():
            results_text += f"  • {emotion.title()}: {count} person{'s' if count > 1 else ''}\n"

        if summary['ages']:
            results_text += f"\n👤 Age Information:\n"
            results_text += f"  • Average Age: {summary['average_age']:.1f} years\n"
            results_text += f"  • Age Range: {min(summary['ages'])} - {max(summary['ages'])} years\n"

        if summary['dress_colors']:
            results_text += "\n👗 Dress Colors:\n"
            for color, count in summary['dress_colors'].items():
                results_text += f"  • {color.title()}: {count} person{'s' if count > 1 else ''}\n"

        results_text += "\n👥 Individual Results:\n"
        for i, detection in enumerate(detections, 1):
            nationality = detection['nationality']
            emotion = detection['emotion']
            age = detection['age']
            dress_color = detection['dress_color']

            person_info = f"Person {i}: {nationality}, {emotion}"
            if age is not None:
                person_info += f", {age} years old"
            if dress_color is not None:
                person_info += f", wearing {dress_color}"

            results_text += f"  • {person_info}\n"

        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", results_text)

        # Update summary labels
        self.people_label.configure(text=f"👥 People: {summary['total_people']}")

        if summary['nationalities']:
            top_nationality = max(summary['nationalities'], key=summary['nationalities'].get)
            self.nationality_label.configure(text=f"🌍 Top Nationality: {top_nationality}")

        if summary['emotions']:
            top_emotion = max(summary['emotions'], key=summary['emotions'].get)
            self.emotion_label.configure(text=f"😊 Top Emotion: {top_emotion.title()}")

        # Enable save buttons
        self.save_results_btn.configure(state="normal")
        self.save_image_btn.configure(state="normal")

    def show_detailed_results_popup(self, detections):
        """Show detailed popup with detection results"""
        popup = ctk.CTkToplevel(self.window)
        popup.title("🌍 Detailed Analysis Results")
        popup.geometry("600x700")
        popup.transient(self.window)
        popup.grab_set()

        # Center the popup
        popup.geometry("+%d+%d" % (
            self.window.winfo_rootx() + 300,
            self.window.winfo_rooty() + 150
        ))

        # Main frame
        main_frame = ctk.CTkFrame(popup)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        ctk.CTkLabel(
            main_frame, 
            text="🎯 Detailed Analysis Results",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=10)

        # Scrollable frame for results
        scrollable_frame = ctk.CTkScrollableFrame(main_frame, height=500)
        scrollable_frame.pack(fill="both", expand=True, pady=10)

        # Display each person's results
        for i, detection in enumerate(detections, 1):
            person_frame = ctk.CTkFrame(scrollable_frame)
            person_frame.pack(fill="x", pady=10, padx=10)

            # Person header
            ctk.CTkLabel(
                person_frame,
                text=f"👤 Person {i}",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=5)

            # Nationality
            nationality_frame = ctk.CTkFrame(person_frame)
            nationality_frame.pack(fill="x", pady=2, padx=10)

            ctk.CTkLabel(
                nationality_frame,
                text=f"🌍 Nationality: {detection['nationality']} ({detection['nationality_confidence']:.2f} confidence)",
                font=ctk.CTkFont(size=12)
            ).pack(pady=2)

            # Emotion
            emotion_frame = ctk.CTkFrame(person_frame)
            emotion_frame.pack(fill="x", pady=2, padx=10)

            emotion_emoji = {"angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😊", 
                           "sad": "😢", "surprise": "😲", "neutral": "😐"}.get(detection['emotion'], "😐")

            ctk.CTkLabel(
                emotion_frame,
                text=f"{emotion_emoji} Emotion: {detection['emotion'].title()} ({detection['emotion_confidence']:.2f} confidence)",
                font=ctk.CTkFont(size=12)
            ).pack(pady=2)

            # Age (conditional)
            if detection['age'] is not None:
                age_frame = ctk.CTkFrame(person_frame)
                age_frame.pack(fill="x", pady=2, padx=10)

                ctk.CTkLabel(
                    age_frame,
                    text=f"👤 Age: {detection['age']} years",
                    font=ctk.CTkFont(size=12)
                ).pack(pady=2)

            # Dress color (conditional)
            if detection['dress_color'] is not None:
                color_frame = ctk.CTkFrame(person_frame)
                color_frame.pack(fill="x", pady=2, padx=10)

                ctk.CTkLabel(
                    color_frame,
                    text=f"👗 Dress Color: {detection['dress_color'].title()}",
                    font=ctk.CTkFont(size=12)
                ).pack(pady=2)

            # Detection rules applied
            rules_frame = ctk.CTkFrame(person_frame)
            rules_frame.pack(fill="x", pady=2, padx=10)

            nationality = detection['nationality']
            if nationality == 'Indian':
                rule_text = "📋 Applied: Nationality + Emotion + Age + Dress Color"
            elif nationality == 'American':
                rule_text = "📋 Applied: Nationality + Emotion + Age"
            elif nationality == 'African':
                rule_text = "📋 Applied: Nationality + Emotion + Dress Color"
            else:
                rule_text = "📋 Applied: Nationality + Emotion Only"

            ctk.CTkLabel(
                rules_frame,
                text=rule_text,
                font=ctk.CTkFont(size=10),
                text_color="gray"
            ).pack(pady=2)

        # Close button
        ctk.CTkButton(
            main_frame,
            text="Close",
            command=popup.destroy
        ).pack(pady=10)

    def save_results(self):
        """Save analysis results to JSON file"""
        if not self.latest_detections:
            messagebox.showwarning("Warning", "No results to save")
            return

        try:
            saved_path = self.detector.save_results(self.current_image_path, self.latest_detections)

            messagebox.showinfo("Success", 
                               f"Analysis results saved to:\n{saved_path}\n\n" +
                               "Contains detailed nationality, emotion, age and dress color data")

        except Exception as e:
            messagebox.showerror("Error", f"Could not save results: {e}")

    def save_annotated_image(self):
        """Save annotated image with detections"""
        if not self.latest_detections:
            messagebox.showwarning("Warning", "No annotated image to save")
            return

        try:
            # Generate annotated image
            annotated_image, _ = self.detector.process_image(self.current_image_path)

            if annotated_image is not None:
                # Ask user where to save
                save_path = filedialog.asksaveasfilename(
                    title="Save Annotated Image",
                    defaultextension=".jpg",
                    filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
                )

                if save_path:
                    cv2.imwrite(save_path, annotated_image)
                    messagebox.showinfo("Success", f"Annotated image saved to:\n{save_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not save annotated image: {e}")

    def run(self):
        """Start the GUI application"""
        self.window.mainloop()

if __name__ == "__main__":
    app = NationalityDetectionGUI()
    app.run()
