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
        self.frame_similarity_threshold = getattr(Config, 'FRAME_SIMILARITY_THRESHOLD', 0.70)
        self.duplicate_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'duplicates')
        
        # Ensure duplicate folder exists
        os.makedirs(self.duplicate_folder, exist_ok=True)
        logger.info(f"DuplicateDetector initialized with similarity threshold: {self.similarity_threshold}, frame threshold: {self.frame_similarity_threshold}")
    
    def detect_duplicates(self, video_id):
        """Detect and flag duplicate videos with timeout protection"""
        start_time = datetime.now()
        timeout_seconds = 15  # Maximum 15 seconds for duplicate detection
        
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
            
            # Limit candidates to prevent timeout (max 5 comparisons)
            if len(candidates) > 5:
                logger.info(f"Limiting duplicate comparison to 5 most recent videos (out of {len(candidates)})")
                candidates = candidates[:5]
            
            # Analyze candidates for duplicates with timeout check
            duplicates = []
            for i, candidate in enumerate(candidates):
                elapsed_time = (datetime.now() - start_time).total_seconds()
                if elapsed_time > timeout_seconds:
                    logger.warning(f"Duplicate detection timeout after {elapsed_time:.1f}s, processed {i}/{len(candidates)} candidates")
                    break
                
                similarity_score = self._calculate_similarity(video, candidate)
                if similarity_score >= self.similarity_threshold:
                    duplicates.append({
                        'video': candidate,
                        'similarity': similarity_score
                    })
            
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
            # Don't raise exception - let processing continue
            logger.info("Continuing with video processing despite duplicate detection error")
    
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
        """Calculate similarity between two videos using enhanced multi-algorithm approach"""
        try:
            logger.info(f"Comparing videos {video1.id} ({video1.filename}) vs {video2.id} ({video2.filename})")
            similarity_factors = []
            
            # 1. Content hash similarity (exact file match) - highest priority
            if video1.video_hash and video2.video_hash:
                hash_similarity = 1.0 if video1.video_hash == video2.video_hash else 0.0
                if hash_similarity == 1.0:
                    logger.info(f"EXACT CONTENT MATCH found between videos {video1.id} and {video2.id}")
                    return 1.0  # Exact duplicate
                similarity_factors.append(hash_similarity * 0.2)  # 20% weight
            
            # 2. Enhanced frame-by-frame perceptual similarity - main detection method
            frame_similarity = self._compare_video_frames(video1.file_path, video2.file_path)
            if frame_similarity is not None:
                # Give high weight to frame similarity and use the specific threshold
                similarity_factors.append(frame_similarity * 0.6)  # 60% weight
                logger.info(f"Frame similarity between videos {video1.id} and {video2.id}: {frame_similarity:.3f}")
                
                # If high frame similarity, likely duplicate even if other factors are low
                if frame_similarity >= self.frame_similarity_threshold:
                    logger.info(f"High frame similarity detected ({frame_similarity:.3f} >= {self.frame_similarity_threshold})")
            
            # 3. Perceptual hash similarity (backup method)
            if video1.perceptual_hash and video2.perceptual_hash:
                perceptual_similarity = self._hamming_similarity(
                    video1.perceptual_hash, video2.perceptual_hash
                )
                similarity_factors.append(perceptual_similarity * 0.1)  # 10% weight
                logger.info(f"Perceptual hash similarity: {perceptual_similarity:.3f}")
            
            # 4. Duration and file size similarity (less weight for embedded videos)
            if video1.duration and video2.duration:
                duration_diff = abs(video1.duration - video2.duration)
                max_duration = max(video1.duration, video2.duration)
                duration_similarity = max(0, 1.0 - (duration_diff / max_duration))
                similarity_factors.append(duration_similarity * 0.1)  # 10% weight
                logger.info(f"Duration similarity: {duration_similarity:.3f} (durations: {video1.duration:.1f}s vs {video2.duration:.1f}s)")
            
            # Calculate final similarity
            final_similarity = sum(similarity_factors) / len(similarity_factors) if similarity_factors else 0.0
            
            # Log detailed results
            logger.info(f"SIMILARITY ANALYSIS for videos {video1.id} vs {video2.id}:")
            logger.info(f"  - Final score: {final_similarity:.3f}")
            logger.info(f"  - Threshold: {self.similarity_threshold}")
            logger.info(f"  - Is duplicate: {'YES' if final_similarity >= self.similarity_threshold else 'NO'}")
            
            return final_similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity between videos {video1.id} and {video2.id}: {str(e)}")
            return 0.0
    
    def _compare_video_frames(self, video_path1, video_path2):
        """Enhanced frame-by-frame comparison with better duplicate detection for embedded videos"""
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
            fps1 = cap1.get(cv2.CAP_PROP_FPS) or 30
            fps2 = cap2.get(cv2.CAP_PROP_FPS) or 30
            
            logger.info(f"Video comparison: {frame_count1} frames vs {frame_count2} frames, FPS: {fps1:.1f} vs {fps2:.1f}")
            
            # Optimized sample size to prevent timeout (up to 10 frames)
            max_samples = 10
            min_samples = 3
            
            # Enhanced sampling strategy - sample from beginning, middle, and end
            frame_similarities = []
            
            # Strategy 1: Direct frame comparison at regular intervals
            if frame_count1 > 0 and frame_count2 > 0:
                # Sample frames from both videos at regular intervals
                samples1 = self._get_frame_sample_positions(frame_count1, max_samples)
                samples2 = self._get_frame_sample_positions(frame_count2, max_samples)
                
                # Optimized comparison - direct frame-to-frame comparison to prevent timeout
                for i, pos1 in enumerate(samples1[:max_samples]):
                    if i >= len(samples2):
                        break
                        
                    cap1.set(cv2.CAP_PROP_POS_FRAMES, pos1)
                    ret1, frame1 = cap1.read()
                    
                    pos2 = samples2[i] if i < len(samples2) else samples2[-1]
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, pos2)
                    ret2, frame2 = cap2.read()
                    
                    if ret1 and ret2:
                        similarity = self._compare_frames(frame1, frame2)
                        if similarity is not None:
                            frame_similarities.append(similarity)
                    
                    # Early exit if we have enough high-similarity matches
                    if len(frame_similarities) >= 5 and np.mean(frame_similarities) > 0.8:
                        logger.info("Early exit: High similarity detected")
                        break
            
            cap1.release()
            cap2.release()
            
            if len(frame_similarities) >= min_samples:
                # Use both average and max similarity for better detection
                avg_similarity = np.mean(frame_similarities)
                max_similarity = np.max(frame_similarities)
                high_similarity_count = len([s for s in frame_similarities if s > 0.8])
                
                # Enhanced scoring: if many frames have high similarity, it's likely a duplicate
                enhanced_score = avg_similarity * 0.7 + max_similarity * 0.3
                if high_similarity_count >= len(frame_similarities) * 0.3:  # 30% of frames highly similar
                    enhanced_score = min(1.0, enhanced_score * 1.2)  # Boost score
                
                logger.info(f"Frame analysis results:")
                logger.info(f"  - Frames compared: {len(frame_similarities)}")
                logger.info(f"  - Average similarity: {avg_similarity:.3f}")
                logger.info(f"  - Max similarity: {max_similarity:.3f}")
                logger.info(f"  - High similarity frames: {high_similarity_count}/{len(frame_similarities)}")
                logger.info(f"  - Enhanced score: {enhanced_score:.3f}")
                
                return enhanced_score
            else:
                logger.warning(f"Insufficient frames compared: {len(frame_similarities)}")
                return None
                
        except Exception as e:
            logger.error(f"Error in enhanced frame comparison: {str(e)}")
            return None
    
    def _get_frame_sample_positions(self, frame_count, max_samples):
        """Get optimized frame positions for fast sampling"""
        if frame_count <= max_samples:
            return list(range(0, frame_count, max(1, frame_count // max_samples)))
        
        # Simple uniform sampling to prevent timeout
        step = frame_count // max_samples
        positions = [i * step for i in range(max_samples)]
        
        # Ensure we don't exceed frame count
        positions = [min(pos, frame_count - 1) for pos in positions]
        
        return positions
    
    def _compare_frames(self, frame1, frame2):
        """Fast frame comparison using optimized perceptual hashing"""
        try:
            # Resize frames to smaller size for faster processing
            height, width = 64, 64  # Reduced size for speed
            frame1_resized = cv2.resize(frame1, (width, height))
            frame2_resized = cv2.resize(frame2, (width, height))
            
            # Convert to PIL Images for perceptual hashing
            frame1_rgb = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB)
            frame2_rgb = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2RGB)
            
            pil_img1 = Image.fromarray(frame1_rgb)
            pil_img2 = Image.fromarray(frame2_rgb)
            
            # Use only one fast hash algorithm to prevent timeout
            try:
                # Use smaller hash size for speed
                hash1 = imagehash.phash(pil_img1, hash_size=8)
                hash2 = imagehash.phash(pil_img2, hash_size=8)
                hash_diff = hash1 - hash2
                max_diff = 8 * 8  # 8x8 hash
                similarity = 1.0 - (hash_diff / max_diff)
                return max(0.0, similarity)
            except Exception as e:
                logger.error(f"Error in frame hash comparison: {str(e)}")
                return None
            
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
