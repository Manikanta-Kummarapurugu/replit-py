#!/usr/bin/env python3
"""
Test script to validate the enhanced duplicate detection system
"""
import os
import sys
from app import app, db
from models import Video
from services.duplicate_detector import DuplicateDetector

def test_duplicate_detection():
    """Test the duplicate detection system"""
    with app.app_context():
        print("=== KrimeWatch Duplicate Detection Test ===")
        
        # Get all videos in the uploads folder
        upload_folder = app.config['UPLOAD_FOLDER']
        video_files = [f for f in os.listdir(upload_folder) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')) 
                      and not f.startswith('DUP_')]
        
        print(f"Found {len(video_files)} video files in uploads folder")
        
        if len(video_files) < 2:
            print("Need at least 2 videos to test duplicate detection")
            return
        
        # Initialize duplicate detector
        detector = DuplicateDetector()
        print(f"Duplicate detector initialized with threshold: {detector.similarity_threshold}")
        
        # Test comparison between first two videos
        video1_path = os.path.join(upload_folder, video_files[0])
        video2_path = os.path.join(upload_folder, video_files[1])
        
        print(f"\nTesting comparison:")
        print(f"Video 1: {video_files[0]}")
        print(f"Video 2: {video_files[1]}")
        
        # Create mock video objects for testing
        class MockVideo:
            def __init__(self, vid_id, filepath, filename):
                self.id = vid_id
                self.file_path = filepath
                self.filename = filename
                self.video_hash = None
                self.perceptual_hash = None
                self.duration = None
        
        mock_video1 = MockVideo(1, video1_path, video_files[0])
        mock_video2 = MockVideo(2, video2_path, video_files[1])
        
        # Test frame comparison
        similarity = detector._compare_video_frames(video1_path, video2_path)
        print(f"\nFrame similarity result: {similarity}")
        
        if similarity is not None:
            if similarity >= detector.frame_similarity_threshold:
                print(f"✅ HIGH SIMILARITY DETECTED ({similarity:.3f} >= {detector.frame_similarity_threshold})")
                print("These videos would be flagged as duplicates!")
            else:
                print(f"❌ Low similarity ({similarity:.3f} < {detector.frame_similarity_threshold})")
                print("These videos would NOT be flagged as duplicates")
        else:
            print("❌ Frame comparison failed")
        
        # Test full similarity calculation
        overall_similarity = detector._calculate_similarity(mock_video1, mock_video2)
        print(f"\nOverall similarity: {overall_similarity:.3f}")
        print(f"Overall threshold: {detector.similarity_threshold}")
        
        if overall_similarity >= detector.similarity_threshold:
            print("✅ DUPLICATE DETECTED!")
        else:
            print("❌ Not a duplicate")

if __name__ == "__main__":
    test_duplicate_detection()