import cv2
import json
import logging
import numpy as np
from datetime import datetime
from app import db
from models import Video, ProcessingLog
from config import Config

logger = logging.getLogger(__name__)

class ContentModerator:
    """Agent for content moderation and inappropriate content detection"""
    
    def __init__(self):
        self.name = "ContentModerator"
        self.inappropriate_flags = Config.INAPPROPRIATE_CONTENT_TYPES
    
    def moderate_video(self, video_id):
        """Moderate video content for inappropriate material"""
        start_time = datetime.now()
        
        try:
            self._log_processing(video_id, 'content_moderation', 'started',
                               'Starting content moderation')
            
            video = Video.query.get(video_id)
            if not video:
                raise ValueError(f"Video {video_id} not found")
            
            # Extract frames for analysis
            frames = self._extract_sample_frames(video.file_path)
            
            if not frames:
                raise ValueError("Could not extract frames from video")
            
            # Analyze content for inappropriate material
            moderation_result = self._analyze_content(frames, video)
            
            # Update video record
            video.is_inappropriate = moderation_result['is_inappropriate']
            video.moderation_flags = json.dumps(moderation_result['flags'])
            
            db.session.commit()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            status = 'flagged' if moderation_result['is_inappropriate'] else 'passed'
            message = f"Content moderation {status}. Flags: {moderation_result['flags']}"
            
            self._log_processing(video_id, 'content_moderation', 'completed',
                               message, processing_time)
            
            return moderation_result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._log_processing(video_id, 'content_moderation', 'failed',
                               f'Error: {str(e)}', processing_time)
            logger.error(f"Error moderating video {video_id}: {str(e)}")
            raise
    
    def _extract_sample_frames(self, file_path, max_frames=10):
        """Extract sample frames for content analysis"""
        try:
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                return []
            
            frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames at regular intervals
            interval = max(1, frame_count // max_frames)
            
            for i in range(0, frame_count, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    frames.append(frame)
                
                if len(frames) >= max_frames:
                    break
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            return []
    
    def _analyze_content(self, frames, video):
        """Analyze video content for inappropriate material"""
        flags = []
        is_inappropriate = False
        
        try:
            # Check video metadata for suspicious indicators
            metadata_flags = self._check_metadata_flags(video)
            flags.extend(metadata_flags)
            
            # Analyze frames for inappropriate content
            frame_flags = self._analyze_frames(frames)
            flags.extend(frame_flags)
            
            # Check for spam/irrelevant content
            spam_flags = self._check_spam_indicators(video, frames)
            flags.extend(spam_flags)
            
            # Determine if video is inappropriate
            is_inappropriate = len(flags) > 0
            
            return {
                'is_inappropriate': is_inappropriate,
                'flags': flags,
                'confidence': min(len(flags) * 0.3, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            return {
                'is_inappropriate': False,
                'flags': [],
                'confidence': 0.0
            }
    
    def _check_metadata_flags(self, video):
        """Check video metadata for suspicious indicators"""
        flags = []
        
        try:
            # Check file size (very small files might be spam)
            if video.file_size < 1024 * 100:  # Less than 100KB
                flags.append('suspicious_file_size')
            
            # Check duration (very short videos might be inappropriate)
            if video.duration and video.duration < 3:
                flags.append('suspicious_duration')
            
            # Check filename for inappropriate keywords
            filename_lower = video.original_filename.lower()
            inappropriate_keywords = [
                'explicit', 'xxx', 'adult', 'porn', 'sex',
                'violence', 'gore', 'death', 'kill',
                'hate', 'racist', 'terrorist'
            ]
            
            for keyword in inappropriate_keywords:
                if keyword in filename_lower:
                    flags.append('inappropriate_filename')
                    break
            
            return flags
            
        except Exception as e:
            logger.error(f"Error checking metadata flags: {str(e)}")
            return []
    
    def _analyze_frames(self, frames):
        """Analyze video frames for inappropriate content"""
        flags = []
        
        try:
            for frame in frames:
                frame_flags = self._analyze_single_frame(frame)
                flags.extend(frame_flags)
            
            # Remove duplicate flags
            return list(set(flags))
            
        except Exception as e:
            logger.error(f"Error analyzing frames: {str(e)}")
            return []
    
    def _analyze_single_frame(self, frame):
        """Analyze a single frame for inappropriate content"""
        flags = []
        
        try:
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Check for excessive motion blur (might indicate violence)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 50:  # Very blurry
                flags.append('excessive_motion_blur')
            
            # Check for unusual color distributions (might indicate inappropriate content)
            color_flags = self._check_color_distribution(hsv)
            flags.extend(color_flags)
            
            # Check for excessive darkness (might hide inappropriate content)
            brightness = np.mean(gray)
            if brightness < 30:  # Very dark
                flags.append('excessive_darkness')
            
            # Check for skin tone detection (basic)
            skin_flags = self._detect_skin_tones(hsv)
            flags.extend(skin_flags)
            
            return flags
            
        except Exception as e:
            logger.error(f"Error analyzing single frame: {str(e)}")
            return []
    
    def _check_color_distribution(self, hsv_frame):
        """Check color distribution for inappropriate content indicators"""
        flags = []
        
        try:
            # Calculate color histograms
            h_hist = cv2.calcHist([hsv_frame], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv_frame], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv_frame], [2], None, [256], [0, 256])
            
            # Check for unusual red dominance (might indicate violence/blood)
            red_range1 = np.sum(h_hist[0:10])
            red_range2 = np.sum(h_hist[170:180])
            total_pixels = hsv_frame.shape[0] * hsv_frame.shape[1]
            red_ratio = (red_range1 + red_range2) / total_pixels
            
            if red_ratio > 0.3:  # More than 30% red
                flags.append('excessive_red_content')
            
            return flags
            
        except Exception as e:
            logger.error(f"Error checking color distribution: {str(e)}")
            return []
    
    def _detect_skin_tones(self, hsv_frame):
        """Detect excessive skin tones (basic adult content detection)"""
        flags = []
        
        try:
            # Define skin tone range in HSV
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            
            # Create mask for skin tones
            skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
            
            # Calculate skin tone ratio
            skin_pixels = cv2.countNonZero(skin_mask)
            total_pixels = hsv_frame.shape[0] * hsv_frame.shape[1]
            skin_ratio = skin_pixels / total_pixels
            
            # Flag if too much skin is detected
            if skin_ratio > 0.4:  # More than 40% skin tones
                flags.append('excessive_skin_content')
            
            return flags
            
        except Exception as e:
            logger.error(f"Error detecting skin tones: {str(e)}")
            return []
    
    def _check_spam_indicators(self, video, frames):
        """Check for spam or irrelevant content indicators"""
        flags = []
        
        try:
            # Check for static images (no motion)
            if len(frames) > 1:
                motion_detected = self._detect_motion(frames)
                if not motion_detected:
                    flags.append('static_content')
            
            # Check for repeated uploads from same source
            # This would require more sophisticated tracking in production
            
            # Check for very low quality (might be auto-generated spam)
            if video.width and video.height:
                if video.width < 320 or video.height < 240:
                    flags.append('low_quality')
            
            return flags
            
        except Exception as e:
            logger.error(f"Error checking spam indicators: {str(e)}")
            return []
    
    def _detect_motion(self, frames):
        """Detect if there's motion between frames"""
        try:
            if len(frames) < 2:
                return True  # Assume motion if only one frame
            
            # Compare consecutive frames
            for i in range(len(frames) - 1):
                gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
                
                # Calculate frame difference
                diff = cv2.absdiff(gray1, gray2)
                motion_pixels = cv2.countNonZero(diff > 30)  # Threshold for motion
                
                total_pixels = gray1.shape[0] * gray1.shape[1]
                motion_ratio = motion_pixels / total_pixels
                
                if motion_ratio > 0.05:  # More than 5% change
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting motion: {str(e)}")
            return True  # Assume motion on error
    
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
