import logging
import math
import os
import shutil
import cv2
import numpy as np
from datetime import datetime, timedelta
from app import db, app
from models import Video, ProcessingLog
from config import Config
import imagehash
from PIL import Image

logger = logging.getLogger(__name__)

class DuplicateDetector:
    """Agent for detecting duplicate videos"""
    
    def __init__(self):
        self.name = "DuplicateDetector"
        self.time_window = Config.DUPLICATE_TIME_WINDOW
        self.distance_radius = Config.DUPLICATE_DISTANCE_RADIUS
        self.similarity_threshold = Config.HASH_SIMILARITY_THRESHOLD
        self.duplicate_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'duplicates')
        
        # Ensure duplicate folder exists
        os.makedirs(self.duplicate_folder, exist_ok=True)
    
    def detect_duplicates(self, video_id):
        """Detect and flag duplicate videos"""
        start_time = datetime.now()
        
        try:
            self._log_processing(video_id, 'duplicate_detection', 'started', 
                               'Starting duplicate detection')
            
            video = Video.query.get(video_id)
            if not video:
                raise ValueError(f"Video {video_id} not found")
            
            # Find potential duplicates
            candidates = self._find_candidate_videos(video)
            
            if not candidates:
                self._log_processing(video_id, 'duplicate_detection', 'completed',
                                   'No duplicate candidates found')
                return
            
            # Analyze candidates for duplicates
            duplicates = self._analyze_candidates(video, candidates)
            
            if duplicates:
                self._handle_duplicates(video, duplicates)
                message = f"Found {len(duplicates)} duplicate(s)"
            else:
                message = "No duplicates detected"
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._log_processing(video_id, 'duplicate_detection', 'completed',
                               message, processing_time)
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._log_processing(video_id, 'duplicate_detection', 'failed',
                               f'Error: {str(e)}', processing_time)
            logger.error(f"Error detecting duplicates for video {video_id}: {str(e)}")
            raise
    
    def _find_candidate_videos(self, video):
        """Find candidate videos for duplicate comparison - Check ALL videos in database"""
        # For robust duplicate detection, we check against ALL videos, not just time/location window
        query = Video.query.filter(
            Video.id != video.id,
            Video.status.in_(['completed', 'processing'])
        )
        
        # Return all videos for comprehensive duplicate checking
        all_candidates = query.all()
        
        logger.info(f"Found {len(all_candidates)} candidate videos to compare against video {video.id}")
        return all_candidates
    
    def _analyze_candidates(self, video, candidates):
        """Analyze candidates for similarity"""
        duplicates = []
        
        for candidate in candidates:
            similarity_score = self._calculate_similarity(video, candidate)
            
            if similarity_score >= self.similarity_threshold:
                duplicates.append({
                    'video': candidate,
                    'similarity': similarity_score
                })
        
        return duplicates
    
    def _calculate_similarity(self, video1, video2):
        """Calculate similarity between two videos using advanced frame-by-frame comparison"""
        try:
            similarity_factors = []
            
            # 1. Content hash similarity (exact file match)
            if video1.video_hash and video2.video_hash:
                hash_similarity = 1.0 if video1.video_hash == video2.video_hash else 0.0
                if hash_similarity == 1.0:
                    logger.info(f"Exact content match found between videos {video1.id} and {video2.id}")
                    return 1.0  # Exact duplicate
                similarity_factors.append(hash_similarity * 0.3)  # 30% weight
            
            # 2. Advanced frame-by-frame perceptual similarity
            frame_similarity = self._compare_video_frames(video1.file_path, video2.file_path)
            if frame_similarity is not None:
                similarity_factors.append(frame_similarity * 0.5)  # 50% weight
                logger.info(f"Frame similarity between videos {video1.id} and {video2.id}: {frame_similarity:.3f}")
            
            # 3. Perceptual hash similarity (backup method)
            if video1.perceptual_hash and video2.perceptual_hash:
                perceptual_similarity = self._hamming_similarity(
                    video1.perceptual_hash, video2.perceptual_hash
                )
                similarity_factors.append(perceptual_similarity * 0.1)  # 10% weight
            
            # 4. Duration similarity
            if video1.duration and video2.duration:
                duration_diff = abs(video1.duration - video2.duration)
                max_duration = max(video1.duration, video2.duration)
                duration_similarity = max(0, 1.0 - (duration_diff / max_duration))
                similarity_factors.append(duration_similarity * 0.1)  # 10% weight
            
            final_similarity = sum(similarity_factors) / len(similarity_factors) if similarity_factors else 0.0
            logger.info(f"Final similarity score between videos {video1.id} and {video2.id}: {final_similarity:.3f}")
            
            return final_similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity between videos {video1.id} and {video2.id}: {str(e)}")
            return 0.0
    
    def _compare_video_frames(self, video_path1, video_path2):
        """Advanced frame-by-frame comparison of two videos"""
        try:
            if not os.path.exists(video_path1) or not os.path.exists(video_path2):
                logger.warning(f"One or both video files not found: {video_path1}, {video_path2}")
                return None
            
            cap1 = cv2.VideoCapture(video_path1)
            cap2 = cv2.VideoCapture(video_path2)
            
            if not cap1.isOpened() or not cap2.isOpened():
                logger.warning(f"Could not open one or both video files for comparison")
                cap1.release()
                cap2.release()
                return None
            
            # Get video properties
            frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            fps1 = cap1.get(cv2.CAP_PROP_FPS)
            fps2 = cap2.get(cv2.CAP_PROP_FPS)
            
            # Sample frames for comparison (up to 20 frames)
            max_samples = 20
            interval1 = max(1, frame_count1 // max_samples)
            interval2 = max(1, frame_count2 // max_samples)
            
            frame_similarities = []
            
            # Compare sampled frames
            for i in range(0, min(frame_count1, frame_count2), max(interval1, interval2)):
                # Get frame from video 1
                cap1.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret1, frame1 = cap1.read()
                
                # Get corresponding frame from video 2
                frame_pos2 = int(i * fps2 / fps1) if fps1 > 0 else i
                cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_pos2)
                ret2, frame2 = cap2.read()
                
                if ret1 and ret2:
                    similarity = self._compare_frames(frame1, frame2)
                    if similarity is not None:
                        frame_similarities.append(similarity)
                
                if len(frame_similarities) >= max_samples:
                    break
            
            cap1.release()
            cap2.release()
            
            if frame_similarities:
                avg_similarity = np.mean(frame_similarities)
                logger.info(f"Compared {len(frame_similarities)} frame pairs, average similarity: {avg_similarity:.3f}")
                return avg_similarity
            else:
                logger.warning("No frames could be compared between videos")
                return None
                
        except Exception as e:
            logger.error(f"Error in frame comparison: {str(e)}")
            return None
    
    def _compare_frames(self, frame1, frame2):
        """Compare two individual frames using perceptual hashing"""
        try:
            # Resize frames to standard size for comparison
            height, width = 64, 64
            frame1_resized = cv2.resize(frame1, (width, height))
            frame2_resized = cv2.resize(frame2, (width, height))
            
            # Convert to PIL Images for perceptual hashing
            frame1_rgb = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB)
            frame2_rgb = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2RGB)
            
            pil_img1 = Image.fromarray(frame1_rgb)
            pil_img2 = Image.fromarray(frame2_rgb)
            
            # Generate perceptual hashes
            hash1 = imagehash.phash(pil_img1)
            hash2 = imagehash.phash(pil_img2)
            
            # Calculate similarity (lower hash difference = higher similarity)
            hash_diff = hash1 - hash2
            max_diff = len(str(hash1)) * 4  # Maximum possible difference
            similarity = 1.0 - (hash_diff / max_diff)
            
            return max(0.0, similarity)
            
        except Exception as e:
            logger.error(f"Error comparing individual frames: {str(e)}")
            return None
    
    def _hamming_similarity(self, hash1, hash2):
        """Calculate Hamming similarity between two hashes"""
        if len(hash1) != len(hash2):
            return 0.0
        
        matches = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matches / len(hash1)
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two GPS coordinates (Haversine formula)"""
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _handle_duplicates(self, video, duplicates):
        """Handle detected duplicates by moving files and updating database"""
        try:
            # Find all videos in the duplicate group (including existing duplicates)
            all_duplicates = [video] + [d['video'] for d in duplicates]
            
            # Find the canonical video (best quality: longest duration, highest resolution, latest upload)
            canonical = self._select_canonical_video(all_duplicates)
            
            # Generate a group ID for this duplicate set
            group_id = f"dup_{canonical.id}_{int(datetime.now().timestamp())}"
            
            duplicates_moved = 0
            
            # Handle each duplicate video
            for dup_video in all_duplicates:
                if dup_video.id != canonical.id:
                    # Move duplicate file to duplicates folder
                    if self._move_duplicate_file(dup_video):
                        dup_video.is_duplicate = True
                        dup_video.canonical_video_id = canonical.id
                        dup_video.duplicate_group_id = group_id
                        duplicates_moved += 1
                        logger.info(f"Moved duplicate video {dup_video.id} ({dup_video.filename}) to duplicates folder")
                    else:
                        logger.error(f"Failed to move duplicate video {dup_video.id} to duplicates folder")
                else:
                    # Mark canonical video with group ID but don't move it
                    dup_video.duplicate_group_id = group_id
                    logger.info(f"Video {canonical.id} ({canonical.filename}) selected as canonical (best quality)")
            
            db.session.commit()
            
            logger.info(f"Processed duplicate group: {duplicates_moved} duplicates moved, canonical video: {canonical.id}")
            
        except Exception as e:
            logger.error(f"Error handling duplicates: {str(e)}")
            db.session.rollback()
            raise
    
    def _select_canonical_video(self, videos):
        """Select the best video as canonical based on quality metrics"""
        def video_quality_score(v):
            score = 0
            # Duration (longer is better)
            score += (v.duration or 0) * 10
            # Resolution (higher is better)
            if v.width and v.height:
                score += (v.width * v.height) / 1000
            # File size (larger often means better quality)
            score += (v.file_size or 0) / (1024 * 1024)  # Convert to MB
            # Upload time (later is often better quality)
            score += v.upload_timestamp.timestamp() / 1000000
            return score
        
        canonical = max(videos, key=video_quality_score)
        logger.info(f"Selected video {canonical.id} as canonical with quality score: {video_quality_score(canonical):.2f}")
        return canonical
    
    def _move_duplicate_file(self, video):
        """Move duplicate video file to duplicates folder"""
        try:
            original_path = video.file_path
            if not os.path.exists(original_path):
                logger.warning(f"Original file not found: {original_path}")
                return False
            
            # Create duplicate filename with timestamp to avoid conflicts
            duplicate_filename = f"DUP_{int(datetime.now().timestamp())}_{video.filename}"
            duplicate_path = os.path.join(self.duplicate_folder, duplicate_filename)
            
            # Move the file
            shutil.move(original_path, duplicate_path)
            
            # Update video record with new path
            video.file_path = duplicate_path
            
            logger.info(f"Successfully moved {original_path} to {duplicate_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving duplicate file: {str(e)}")
            return False
    
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
