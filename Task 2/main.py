#!/usr/bin/env python3
"""
Animal Detection System - Main Entry Point
Run this file to start the GUI application
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from animal_detection_gui import AnimalDetectionGUI

    def main():
        """Main function to start the application"""
        print("🦁 Starting Animal Detection System...")
        print("=" * 50)
        print("Features:")
        print("• Multi-animal detection in images and videos")
        print("• Species classification (80+ animals)")
        print("• Carnivore highlighting in red")
        print("• Real-time video processing")
        print("• Pop-up alerts for carnivorous animals")
        print("• Export detection results")
        print("=" * 50)

        # Initialize and run GUI
        app = AnimalDetectionGUI()
        app.run()

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please install required dependencies:")
    print("pip install -r requirements.txt")
    print()
    print("Required packages:")
    print("• ultralytics (YOLOv8)")
    print("• opencv-python")
    print("• customtkinter")
    print("• pillow")
    print("• torch")

except Exception as e:
    print(f"❌ Error starting application: {e}")
    print("Please check your installation and try again.")
