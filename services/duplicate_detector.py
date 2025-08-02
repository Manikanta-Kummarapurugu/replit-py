import logging
import math
from datetime import datetime, timedelta
from app import db
from models import Video, ProcessingLog
from config import Config

logger = logging.getLogger(__name__)

class DuplicateDetector:
    """Agent for detecting duplicate videos"""
    
    def __init__(self):
        self.name = "DuplicateDetector"
        self.time_window = Config.DUPLICATE_TIME_WINDOW
        self.distance_radius = Config.DUPLICATE_DISTANCE_RADIUS
        self.similarity_threshold = Config.HASH_SIMILARITY_THRESHOLD
    
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
        """Find candidate videos for duplicate comparison"""
        # Calculate time window
        time_start = video.upload_timestamp - timedelta(seconds=self.time_window)
        time_end = video.upload_timestamp + timedelta(seconds=self.time_window)
        
        # Base query for time window
        query = Video.query.filter(
            Video.id != video.id,
            Video.upload_timestamp.between(time_start, time_end),
            Video.status.in_(['completed', 'processing'])
        )
        
        # If GPS coordinates are available, filter by location
        if video.gps_latitude and video.gps_longitude:
            candidates = []
            all_videos = query.all()
            
            for candidate in all_videos:
                if candidate.gps_latitude and candidate.gps_longitude:
                    distance = self._calculate_distance(
                        video.gps_latitude, video.gps_longitude,
                        candidate.gps_latitude, candidate.gps_longitude
                    )
                    if distance <= self.distance_radius:
                        candidates.append(candidate)
                else:
                    # Include videos without GPS if they match other criteria
                    candidates.append(candidate)
            
            return candidates
        else:
            # If no GPS, return all videos in time window
            return query.all()
    
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
        """Calculate similarity between two videos"""
        similarity_factors = []
        
        # Hash similarity (most important)
        if video1.video_hash and video2.video_hash:
            hash_similarity = 1.0 if video1.video_hash == video2.video_hash else 0.0
            similarity_factors.append(hash_similarity * 0.6)  # 60% weight
        
        # Perceptual hash similarity
        if video1.perceptual_hash and video2.perceptual_hash:
            perceptual_similarity = self._hamming_similarity(
                video1.perceptual_hash, video2.perceptual_hash
            )
            similarity_factors.append(perceptual_similarity * 0.3)  # 30% weight
        
        # Duration similarity
        if video1.duration and video2.duration:
            duration_diff = abs(video1.duration - video2.duration)
            max_duration = max(video1.duration, video2.duration)
            duration_similarity = 1.0 - (duration_diff / max_duration)
            similarity_factors.append(duration_similarity * 0.1)  # 10% weight
        
        return sum(similarity_factors) / len(similarity_factors) if similarity_factors else 0.0
    
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
        """Handle detected duplicates"""
        # Find all videos in the duplicate group (including existing duplicates)
        all_duplicates = [video] + [d['video'] for d in duplicates]
        
        # Find the canonical video (longest duration, or latest if same duration)
        canonical = max(all_duplicates, key=lambda v: (v.duration or 0, v.upload_timestamp))
        
        # Generate a group ID for this duplicate set
        group_id = f"dup_{canonical.id}_{int(datetime.now().timestamp())}"
        
        # Mark all videos as duplicates except the canonical one
        for dup_video in all_duplicates:
            if dup_video.id != canonical.id:
                dup_video.is_duplicate = True
                dup_video.canonical_video_id = canonical.id
                dup_video.duplicate_group_id = group_id
            else:
                dup_video.duplicate_group_id = group_id
        
        db.session.commit()
        
        logger.info(f"Marked {len(all_duplicates)-1} videos as duplicates of canonical video {canonical.id}")
    
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
