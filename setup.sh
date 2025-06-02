#!/bin/bash

# Face Duplicate Detection Setup Script
# ====================================

echo "🚀 Starting Face Duplicate Detection Process..."
echo "==========================================================================================="

# Step 1: Run face recognition
echo ""
echo "📸 Step 1: Running face recognition..."
echo "--------------------------------------"
python face_recognition.py

# Check if face_recognition.py executed successfully
if [ $? -eq 0 ]; then
    echo "✅ Face recognition completed successfully!"
else
    echo "❌ Face recognition failed!"
    exit 1
fi
echo "=========================================================================================="
# Step 2: Run duplicate face detection
echo ""
echo "🔍 Step 2: Running duplicate face detection..."
echo "--------------------------------------------"
python duplicate_face.py

# Check if duplicate_face.py executed successfully
if [ $? -eq 0 ]; then
    echo "✅ Duplicate face detection completed successfully!"
else
    echo "❌ Duplicate face detection failed!"
    exit 1
fi

echo ""
echo "🎉 Face Duplicate Detection Process Complete!"
echo "=========================================================================================="