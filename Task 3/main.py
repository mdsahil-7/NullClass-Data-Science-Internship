#!/usr/bin/env python3
"""
Drowsiness Detection System - Main Entry Point
Vehicle Safety Monitoring System
Run this file to start the GUI application
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from drowsiness_detection_gui import DrowsinessDetectionGUI

    def main():
        """Main function to start the application"""
        print("😴 Starting Drowsiness Detection System...")
        print("=" * 60)
        print("🚗 Vehicle Safety Monitoring System")
        print("=" * 60)
        print("Features:")
        print("• Multi-person drowsiness detection")
        print("• Real-time age estimation")
        print("• Vehicle safety monitoring")
        print("• Sleep detection with red highlighting")
        print("• Pop-up alerts for sleeping passengers")
        print("• Video and image processing")
        print("• Emergency alert system")
        print("=" * 60)
        print("🎯 Detection Methods:")
        print("• Eye Aspect Ratio (EAR) analysis")
        print("• Facial landmark detection")
        print("• Multiple face detection algorithms")
        print("• ML-based age estimation")
        print("=" * 60)

        # Initialize and run GUI
        try:
            app = DrowsinessDetectionGUI()
            print("✅ GUI initialized successfully!")
            print("🚀 Launching application...")
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
    print("• tensorflow (Deep Learning)")
    print("• dlib (Facial Landmarks)")
    print("• customtkinter (Modern GUI)")
    print("• face-recognition (Face Detection)")
    print("• mediapipe (Face Processing)")
    print("\n⚠️ Additional Setup Required:")
    print("• Download dlib shape predictor:")
    print("  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print("  Extract and place in models/ directory")

except Exception as e:
    print(f"❌ Error starting application: {e}")
    print("\n🔧 Troubleshooting:")
    print("• Ensure all dependencies are installed")
    print("• Check camera/webcam permissions")
    print("• Verify file paths are correct")
    print("• Try running with administrator privileges")
    print("\n📞 For support, check README.md")
