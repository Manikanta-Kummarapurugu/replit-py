import os
import cv2
import hashlib
import logging
from datetime import datetime
from app import db
from models import Video, ProcessingLog
import imagehash
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Agent for video ingestion and metadata extraction"""
    
    def __init__(self):
        self.name = "VideoProcessor"
    
    def process_video(self, video_id):
        """Process video and extract metadata"""
        start_time = datetime.now()
        
        try:
            # Log processing start
            self._log_processing(video_id, 'ingestion', 'started', 'Starting video ingestion')
            
            video = Video.query.get(video_id)
            if not video:
                raise ValueError(f"Video {video_id} not found")
            
            # Extract video metadata using OpenCV
            metadata = self._extract_video_metadata(video.file_path)
            
            # Generate video hashes
            hashes = self._generate_video_hashes(video.file_path)
            
            # Update video record with metadata
            video.duration = metadata.get('duration')
            video.width = metadata.get('width')
            video.height = metadata.get('height')
            video.fps = metadata.get('fps')
            video.video_hash = hashes.get('content_hash')
            video.perceptual_hash = hashes.get('perceptual_hash')
            
            db.session.commit()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._log_processing(video_id, 'ingestion', 'completed', 
                               f'Video metadata extracted successfully', processing_time)
            
            logger.info(f"Video {video_id} processed successfully")
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._log_processing(video_id, 'ingestion', 'failed', 
                               f'Error: {str(e)}', processing_time)
            logger.error(f"Error processing video {video_id}: {str(e)}")
            raise
    
    def _extract_video_metadata(self, file_path):
        """Extract video metadata using OpenCV"""
        try:
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'duration': duration,
                'fps': fps,
                'width': width,
                'height': height,
                'frame_count': frame_count
            }
            
        except Exception as e:
            logger.error(f"Error extracting video metadata: {str(e)}")
            return {}
    
    def _generate_video_hashes(self, file_path):
        """Generate content and perceptual hashes for the video"""
        try:
            hashes = {}
            
            # Generate content hash (MD5 of file)
            with open(file_path, 'rb') as f:
                content = f.read()
                hashes['content_hash'] = hashlib.md5(content).hexdigest()
            
            # Generate perceptual hash using video frames
            hashes['perceptual_hash'] = self._generate_perceptual_hash(file_path)
            
            return hashes
            
        except Exception as e:
            logger.error(f"Error generating video hashes: {str(e)}")
            return {}
    
    def _generate_perceptual_hash(self, file_path):
        """Generate perceptual hash from video frames"""
        try:
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                return None
            
            frame_hashes = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames at regular intervals (max 10 frames)
            interval = max(1, frame_count // 10)
            
            for i in range(0, frame_count, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    # Convert to PIL Image for hashing
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Generate perceptual hash
                    frame_hash = str(imagehash.phash(pil_image))
                    frame_hashes.append(frame_hash)
                
                if len(frame_hashes) >= 10:
                    break
            
            cap.release()
            
            # Combine frame hashes into a single hash
            combined = ''.join(frame_hashes)
            return hashlib.sha256(combined.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating perceptual hash: {str(e)}")
            return None
    
    def _log_processing(self, video_id, step, status, message, processing_time=None):
        """Log processing step"""
        log = ProcessingLog(
            video_id=video_id,
            step=step,
            status=status,
            message=message,
            processing_time=processing_time
        )
        db.session.add(log)
        db.session.commit()
