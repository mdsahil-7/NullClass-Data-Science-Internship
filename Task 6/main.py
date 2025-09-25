#!/usr/bin/env python3
"""
Car Color Detection & Traffic Analysis System - Main Entry Point
AI-Powered Traffic Monitoring and Vehicle Color Recognition
Run this file to start the GUI application
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from car_color_detection_gui import CarColorDetectionGUI

    def main():
        """Main function to start the application"""
        print("🚗 Starting Car Color Detection & Traffic Analysis System...")
        print("=" * 80)
        print("🚦 AI-Powered Traffic Monitoring System")
        print("=" * 80)
        print("Features:")
        print("• 🎨 Car color detection (7+ colors)")
        print("• 🚗 Automatic car counting")
        print("• 👥 People detection and counting")
        print("• 🔴 Red rectangles for blue cars")
        print("• 🔵 Blue rectangles for other colored cars")
        print("• 🟢 Green rectangles for people")
        print("• 📊 Real-time traffic statistics")
        print("• 💾 Results export functionality")
        print("=" * 80)
        print("🎯 Detection Capabilities:")
        print("• Vehicle Types: Cars, Trucks, Buses, Motorcycles")
        print("• Car Colors: Blue, Red, Green, Yellow, White, Black, Gray")
        print("• Object Detection: YOLO-based detection system")
        print("• Color Analysis: K-means clustering + HSV analysis")
        print("• People Detection: Full body detection")
        print("=" * 80)
        print("📊 Rectangle Color Coding:")
        print("🔴 RED RECTANGLES → Blue Cars")
        print("🔵 BLUE RECTANGLES → All Other Colored Cars")
        print("🟢 GREEN RECTANGLES → People")
        print("=" * 80)
        print("🎮 Usage Instructions:")
        print("1. Upload Image: Select traffic scene image")
        print("2. Analyze Traffic: Click analysis button")
        print("3. View Results: See colored rectangles and counts")
        print("4. Check Statistics: Total cars, blue cars, people count")
        print("5. Save Results: Export analysis data and images")
        print("=" * 80)

        # Initialize and run GUI
        try:
            app = CarColorDetectionGUI()
            print("✅ GUI initialized successfully!")
            print("🚀 Launching Car Color Detection System...")
            print("\n📝 Note: For best results, use clear traffic scene images")
            print("🚦 System supports traffic intersections, parking lots, and roadways")
            app.run()
        except Exception as e:
            print(f"❌ Error launching GUI: {e}")
            print("Please check your installation and dependencies.")

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nPlease install required dependencies:")
    print("pip install -r requirements.txt")
    print("\n📦 Key Dependencies:")
    print("• opencv-python (Computer Vision)")
    print("• ultralytics (YOLOv8 Object Detection)")
    print("• tensorflow (Deep Learning)")
    print("• torch/torchvision (PyTorch)")
    print("• scikit-learn (K-means Clustering)")
    print("• customtkinter (Modern GUI)")
    print("• webcolors (Color Name Mapping)")
    print("• colorthief (Color Extraction)")
    print("\n🚗 Additional Requirements:")
    print("• YOLO model will download automatically (~6MB)")
    print("• Good quality traffic images recommended")
    print("• Multiple car colors for best demonstration")

except Exception as e:
    print(f"❌ Error starting application: {e}")
    print("\n🔧 Troubleshooting:")
    print("• Ensure all dependencies are installed")
    print("• Check internet connection (for YOLO model download)")
    print("• Verify image file formats are supported")
    print("• Try running with administrator privileges")
    print("• Ensure sufficient disk space for models")
    print("\n📞 For support, check README.md")
