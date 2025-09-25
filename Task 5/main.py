#!/usr/bin/env python3
"""
Sign Language Detection System - Main Entry Point
AI-Powered ASL Recognition with Time-Based Operation
Run this file to start the GUI application
"""

import sys
import os
from datetime import datetime, time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sign_language_detection_gui import SignLanguageDetectionGUI

    def main():
        """Main function to start the application"""
        print("🤟 Starting Sign Language Detection System...")
        print("=" * 70)
        print("🎯 AI-Powered American Sign Language (ASL) Recognition")
        print("=" * 70)
        print("Features:")
        print("• 🤟 Real-time ASL sign recognition")
        print("• 📹 Live camera feed processing")
        print("• 🖼️ Static image analysis")
        print("• ⏰ Time-based operation (6:00 PM - 10:00 PM)")
        print("• 🗣️ Text-to-speech output")
        print("• 🤲 Multi-hand detection")
        print("• 📊 Confidence scoring")
        print("=" * 70)
        print("🤟 Supported ASL Signs:")
        print("Hello, Thank You, Please, Yes, No, Good, Bad,")
        print("Help, Water, Food, Love, Peace, Stop, Go, Come,")
        print("Beautiful, Family, Friend, Home, Work")
        print("=" * 70)
        print("⏰ Operation Schedule:")

        # Check current operation status
        current_time = datetime.now().time()
        operation_start = time(18, 0)  # 6:00 PM
        operation_end = time(22, 0)    # 10:00 PM

        is_active = operation_start <= current_time <= operation_end
        current_time_str = datetime.now().strftime('%I:%M %p')

        print(f"Current Time: {current_time_str}")
        print(f"Active Hours: {operation_start.strftime('%I:%M %p')} - {operation_end.strftime('%I:%M %p')}")

        if is_active:
            print("🟢 STATUS: SYSTEM ACTIVE - Ready for detection")
        else:
            print("🔴 STATUS: SYSTEM INACTIVE - Outside operation hours")
            print("   Camera and detection features will be limited")

        print("=" * 70)
        print("🎮 Usage Instructions:")
        print("1. Upload Image: Select image file for sign detection")
        print("2. Start Camera: Real-time ASL recognition (operation hours only)")
        print("3. View Results: See detected signs with confidence scores")
        print("4. Text-to-Speech: Hear detected signs spoken aloud")
        print("5. Save Results: Export detection data and screenshots")
        print("=" * 70)

        # Initialize and run GUI
        try:
            app = SignLanguageDetectionGUI()
            print("✅ GUI initialized successfully!")
            print("🚀 Launching Sign Language Detection System...")
            print("\n📝 Note: For best results, ensure good lighting and clear hand gestures")
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
    print("• mediapipe (Hand Detection)")
    print("• customtkinter (Modern GUI)")
    print("• pyttsx3 (Text-to-Speech)")
    print("• pillow (Image Processing)")
    print("• scikit-learn (Machine Learning)")
    print("\n🤟 System Requirements:")
    print("• Webcam (for real-time detection)")
    print("• Microphone/Speakers (for audio output)")
    print("• 4GB+ RAM (for model processing)")
    print("• Good lighting (for hand detection)")

except Exception as e:
    print(f"❌ Error starting application: {e}")
    print("\n🔧 Troubleshooting:")
    print("• Ensure webcam is connected and accessible")
    print("• Check system time for operation hours")
    print("• Verify all dependencies are installed")
    print("• Try running with administrator privileges")
    print("• Ensure good lighting for hand detection")
    print("\n📞 For support, check README.md")
