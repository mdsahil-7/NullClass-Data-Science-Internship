#!/usr/bin/env python3
"""
Nationality & Emotion Detection System - Main Entry Point
AI-Powered Multi-Cultural Analysis System
Run this file to start the GUI application
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from nationality_detection_gui import NationalityDetectionGUI

    def main():
        """Main function to start the application"""
        print("🌍 Starting Nationality & Emotion Detection System...")
        print("=" * 70)
        print("🎯 AI-Powered Multi-Cultural Analysis System")
        print("=" * 70)
        print("Features:")
        print("• 🌍 Nationality Detection (9+ nationalities)")
        print("• 😊 Emotion Recognition (7 emotions)")
        print("• 👤 Age Estimation (conditional)")
        print("• 👗 Dress Color Detection (conditional)")
        print("• 🎨 Color-coded nationality visualization")
        print("• 📊 Detailed analysis reports")
        print("=" * 70)
        print("🔧 Detection Rules:")
        print("🇮🇳 Indian: Nationality + Emotion + Age + Dress Color")
        print("🇺🇸 American: Nationality + Emotion + Age")  
        print("🌍 African: Nationality + Emotion + Dress Color")
        print("🌎 Others: Nationality + Emotion Only")
        print("=" * 70)
        print("📊 Supported Nationalities:")
        print("Indian, American, African, Chinese, European,")
        print("Middle Eastern, East Asian, Latino, Other")
        print("=" * 70)

        # Initialize and run GUI
        try:
            app = NationalityDetectionGUI()
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
    print("• face-recognition (Face Detection)")
    print("• customtkinter (Modern GUI)")
    print("• mediapipe (Face Processing)")
    print("• scikit-learn (Machine Learning)")
    print("• colorthief (Color Analysis)")
    print("• webcolors (Color Name Mapping)")
    print("\n🎯 Optional Enhancements:")
    print("• Download trained models for better accuracy")
    print("• Add custom nationality datasets")
    print("• Configure emotion recognition models")

except Exception as e:
    print(f"❌ Error starting application: {e}")
    print("\n🔧 Troubleshooting:")
    print("• Ensure all dependencies are installed")
    print("• Check camera/webcam permissions")
    print("• Verify file paths are correct")
    print("• Try running with administrator privileges")
    print("\n📞 For support, check README.md")
