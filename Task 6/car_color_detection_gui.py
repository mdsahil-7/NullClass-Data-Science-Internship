import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
from car_color_detector import CarColorDetector
from datetime import datetime

class CarColorDetectionGUI:
    def __init__(self):
        # Initialize CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.window = ctk.CTk()
        self.window.title("🚗 Car Color Detection & Traffic Analysis System")
        self.window.geometry("1600x1000")

        self.detector = CarColorDetector()
        self.current_image_path = None
        self.latest_car_detections = []
        self.latest_people_detections = []

        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI layout"""
        # Main title
        title_label = ctk.CTkLabel(
            self.window, 
            text="🚗 Car Color Detection & Traffic Analysis System",
            font=ctk.CTkFont(size=26, weight="bold")
        )
        title_label.pack(pady=20)

        subtitle_label = ctk.CTkLabel(
            self.window,
            text="AI-Powered Traffic Monitoring: Car Colors, Counting & People Detection",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        subtitle_label.pack(pady=(0, 20))

        # Main container
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Left panel - Controls
        left_panel = ctk.CTkFrame(main_frame, width=420)
        left_panel.pack(side="left", fill="y", padx=10, pady=10)
        left_panel.pack_propagate(False)

        # File selection section
        file_frame = ctk.CTkFrame(left_panel)
        file_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(file_frame, text="📁 Traffic Image Selection", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.select_image_btn = ctk.CTkButton(
            file_frame, text="🖼️ Upload Traffic Image", 
            command=self.select_image, height=40
        )
        self.select_image_btn.pack(pady=5, padx=10, fill="x")

        # Detection controls
        detection_frame = ctk.CTkFrame(left_panel)
        detection_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(detection_frame, text="🎯 Analysis Controls", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.analyze_btn = ctk.CTkButton(
            detection_frame, text="🚦 Analyze Traffic Scene", 
            command=self.analyze_traffic, height=50, state="disabled",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.analyze_btn.pack(pady=10, padx=10, fill="x")

        # Detection rules info
        rules_frame = ctk.CTkFrame(left_panel)
        rules_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(rules_frame, text="📋 Detection Rules", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        rules_text = ctk.CTkTextbox(rules_frame, height=100)
        rules_text.pack(fill="x", padx=10, pady=5)

        rules_content = """🔴 Blue Cars: Red Rectangles
🔵 Other Cars: Blue Rectangles  
🟢 People: Green Rectangles
📊 Automatic counting of all objects
🎨 7 car colors detected: Blue, Red, Green, Yellow, White, Black, Gray"""

        rules_text.insert("1.0", rules_content)
        rules_text.configure(state="disabled")

        # Statistics panel
        stats_frame = ctk.CTkFrame(left_panel)
        stats_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(stats_frame, text="📊 Traffic Statistics", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Statistics displays
        self.total_cars_label = ctk.CTkLabel(
            stats_frame, text="🚗 Total Cars: 0", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.total_cars_label.pack(pady=2)

        self.blue_cars_label = ctk.CTkLabel(
            stats_frame, text="🔵 Blue Cars: 0", 
            font=ctk.CTkFont(size=12, weight="bold"), text_color="lightblue"
        )
        self.blue_cars_label.pack(pady=1)

        self.other_cars_label = ctk.CTkLabel(
            stats_frame, text="🚗 Other Cars: 0", 
            font=ctk.CTkFont(size=12, weight="bold"), text_color="lightgreen"
        )
        self.other_cars_label.pack(pady=1)

        self.people_label = ctk.CTkLabel(
            stats_frame, text="👥 People: 0", 
            font=ctk.CTkFont(size=12, weight="bold"), text_color="yellow"
        )
        self.people_label.pack(pady=1)

        self.confidence_label = ctk.CTkLabel(
            stats_frame, text="📊 Avg Confidence: 0%", 
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.confidence_label.pack(pady=1)

        # Results panel
        results_frame = ctk.CTkFrame(left_panel)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(results_frame, text="📋 Detailed Results", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Results text area
        self.results_text = ctk.CTkTextbox(results_frame, height=200)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Right panel - Preview and Results
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(right_panel, text="🖥️ Traffic Scene Preview & Analysis", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Image display
        self.display_frame = ctk.CTkFrame(right_panel, fg_color="gray20")
        self.display_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_label = ctk.CTkLabel(
            self.display_frame, 
            text="No traffic image selected\n\n🚗 Upload a traffic scene image\n🔴 Blue cars → Red rectangles\n🔵 Other cars → Blue rectangles\n🟢 People → Green rectangles", 
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

        # Color legend
        legend_frame = ctk.CTkFrame(control_frame)
        legend_frame.pack(side="top", pady=5)

        legend_text = ctk.CTkLabel(
            legend_frame,
            text="Legend: 🔴 Blue Cars | 🔵 Other Cars | 🟢 People",
            font=ctk.CTkFont(size=12)
        )
        legend_text.pack(pady=5)

    def select_image(self):
        """Select traffic image file"""
        file_path = filedialog.askopenfilename(
            title="Select Traffic Image",
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
            self.results_text.insert("end", "Click 'Analyze Traffic Scene' to detect cars and people...\n")

            # Reset statistics
            self.reset_statistics()

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

    def reset_statistics(self):
        """Reset all statistics displays"""
        self.total_cars_label.configure(text="🚗 Total Cars: 0")
        self.blue_cars_label.configure(text="🔵 Blue Cars: 0")
        self.other_cars_label.configure(text="🚗 Other Cars: 0")
        self.people_label.configure(text="👥 People: 0")
        self.confidence_label.configure(text="📊 Avg Confidence: 0%")

    def analyze_traffic(self):
        """Run traffic analysis on selected image"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return

        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", "🚦 Analyzing traffic scene...\n")
        self.results_text.insert("end", "🚗 Detecting cars and analyzing colors...\n")
        self.results_text.insert("end", "👥 Detecting people...\n")
        self.window.update()

        def analysis_thread():
            try:
                # Run traffic analysis
                annotated_image, car_detections, people_detections, blue_cars, other_cars = self.detector.process_traffic_image(self.current_image_path)

                if annotated_image is not None:
                    self.latest_car_detections = car_detections
                    self.latest_people_detections = people_detections

                    # Display results
                    self.display_analysis_results(annotated_image, car_detections, people_detections, blue_cars, other_cars)

                    # Show detailed results popup
                    if car_detections or people_detections:
                        self.show_detailed_results_popup(car_detections, people_detections)

                else:
                    self.results_text.insert("end", "❌ Analysis failed - could not process image\n")

            except Exception as e:
                self.results_text.insert("end", f"❌ Error during analysis: {e}\n")

        thread = threading.Thread(target=analysis_thread)
        thread.daemon = True
        thread.start()

    def display_analysis_results(self, annotated_image, car_detections, people_detections, blue_cars, other_cars):
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

        # Update statistics
        summary = self.detector.get_detection_summary(car_detections, people_detections)

        self.total_cars_label.configure(text=f"🚗 Total Cars: {summary['total_cars']}")
        self.blue_cars_label.configure(text=f"🔵 Blue Cars: {summary['blue_cars']}")
        self.other_cars_label.configure(text=f"🚗 Other Cars: {summary['other_cars']}")
        self.people_label.configure(text=f"👥 People: {summary['total_people']}")
        self.confidence_label.configure(text=f"📊 Avg Confidence: {summary['average_confidence']*100:.1f}%")

        # Update results text
        results_text = f"""
🚦 Traffic Analysis Complete!

📊 Detection Summary:
• Total Vehicles: {summary['total_cars']}
• Blue Cars (Red boxes): {summary['blue_cars']}
• Other Cars (Blue boxes): {summary['other_cars']}
• People Detected: {summary['total_people']}
• Average Confidence: {summary['average_confidence']*100:.1f}%

🎨 Car Colors Detected:
"""

        if summary['car_colors']:
            for color, count in summary['car_colors'].items():
                results_text += f"• {color.title()}: {count} car{'s' if count > 1 else ''}\n"
        else:
            results_text += "• No cars detected\n"

        results_text += "\n🚗 Individual Car Results:\n"
        for i, car in enumerate(car_detections, 1):
            box_color = "🔴 RED BOX" if car['color'].lower() == 'blue' else "🔵 BLUE BOX"
            results_text += f"• Car {i}: {car['color'].title()} - {box_color} ({car['confidence']:.2f})\n"

        if people_detections:
            results_text += "\n👥 People Results:\n"
            for i, person in enumerate(people_detections, 1):
                results_text += f"• Person {i}: 🟢 GREEN BOX ({person['confidence']:.2f})\n"

        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", results_text)

        # Enable save buttons
        self.save_results_btn.configure(state="normal")
        self.save_image_btn.configure(state="normal")

    def show_detailed_results_popup(self, car_detections, people_detections):
        """Show detailed popup with detection results"""
        popup = ctk.CTkToplevel(self.window)
        popup.title("🚦 Detailed Traffic Analysis")
        popup.geometry("650x700")
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
            text="🚦 Detailed Traffic Analysis Results",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=10)

        # Scrollable frame for results
        scrollable_frame = ctk.CTkScrollableFrame(main_frame, height=500)
        scrollable_frame.pack(fill="both", expand=True, pady=10)

        # Car detection results
        if car_detections:
            ctk.CTkLabel(
                scrollable_frame,
                text="🚗 Car Detection Results",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=(0, 10))

            for i, car in enumerate(car_detections, 1):
                car_frame = ctk.CTkFrame(scrollable_frame)
                car_frame.pack(fill="x", pady=5, padx=10)

                # Car info
                color = car['color']
                box_color = "Red Rectangle" if color.lower() == 'blue' else "Blue Rectangle"
                box_emoji = "🔴" if color.lower() == 'blue' else "🔵"

                car_info = f"{box_emoji} Car {i}: {color.title()} Color - {box_color}"

                ctk.CTkLabel(
                    car_frame,
                    text=car_info,
                    font=ctk.CTkFont(size=14, weight="bold")
                ).pack(pady=5)

                # Technical details
                ctk.CTkLabel(
                    car_frame,
                    text=f"Confidence: {car['confidence']:.3f} | Class: {car['class_name'].title()}",
                    font=ctk.CTkFont(size=12),
                    text_color="gray"
                ).pack(pady=2)

        # People detection results
        if people_detections:
            ctk.CTkLabel(
                scrollable_frame,
                text="👥 People Detection Results",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=(20, 10))

            for i, person in enumerate(people_detections, 1):
                person_frame = ctk.CTkFrame(scrollable_frame)
                person_frame.pack(fill="x", pady=5, padx=10)

                # Person info
                ctk.CTkLabel(
                    person_frame,
                    text=f"🟢 Person {i}: Green Rectangle",
                    font=ctk.CTkFont(size=14, weight="bold")
                ).pack(pady=5)

                # Technical details
                ctk.CTkLabel(
                    person_frame,
                    text=f"Confidence: {person['confidence']:.3f}",
                    font=ctk.CTkFont(size=12),
                    text_color="gray"
                ).pack(pady=2)

        # Summary statistics
        summary_frame = ctk.CTkFrame(scrollable_frame)
        summary_frame.pack(fill="x", pady=10, padx=10)

        ctk.CTkLabel(
            summary_frame,
            text="📊 Traffic Summary",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)

        summary = self.detector.get_detection_summary(car_detections, people_detections)

        summary_text = f"""Total Cars: {summary['total_cars']}
Blue Cars (Red Boxes): {summary['blue_cars']}
Other Cars (Blue Boxes): {summary['other_cars']}
People (Green Boxes): {summary['total_people']}
Average Detection Confidence: {summary['average_confidence']*100:.1f}%"""

        ctk.CTkLabel(
            summary_frame,
            text=summary_text,
            font=ctk.CTkFont(size=12)
        ).pack(pady=5)

        # Close button
        ctk.CTkButton(
            main_frame,
            text="Close",
            command=popup.destroy
        ).pack(pady=10)

    def save_results(self):
        """Save analysis results to JSON file"""
        if not self.latest_car_detections and not self.latest_people_detections:
            messagebox.showwarning("Warning", "No results to save")
            return

        try:
            saved_path = self.detector.save_results(
                self.current_image_path, 
                self.latest_car_detections, 
                self.latest_people_detections
            )

            messagebox.showinfo("Success", 
                               f"Traffic analysis results saved to:\n{saved_path}\n\n" +
                               "Contains car colors, counts, people detection, and bounding boxes")

        except Exception as e:
            messagebox.showerror("Error", f"Could not save results: {e}")

    def save_annotated_image(self):
        """Save annotated image with detections"""
        if not self.latest_car_detections and not self.latest_people_detections:
            messagebox.showwarning("Warning", "No annotated image to save")
            return

        try:
            # Generate annotated image
            annotated_image, _, _, _, _ = self.detector.process_traffic_image(self.current_image_path)

            if annotated_image is not None:
                # Ask user where to save
                save_path = filedialog.asksaveasfilename(
                    title="Save Annotated Traffic Image",
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
    app = CarColorDetectionGUI()
    app.run()
