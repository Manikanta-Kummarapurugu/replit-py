from app import db
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean

class Video(db.Model):
    __tablename__ = 'videos'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    duration = Column(Float, nullable=True)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Metadata
    video_hash = Column(String(128), nullable=True)
    perceptual_hash = Column(String(128), nullable=True)
    gps_latitude = Column(Float, nullable=True)
    gps_longitude = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    
    # Processing status
    status = Column(String(50), default='processing')  # processing, completed, failed
    is_duplicate = Column(Boolean, default=False)
    canonical_video_id = Column(Integer, nullable=True)
    duplicate_group_id = Column(String(128), nullable=True)
    
    # Classification results
    classification = Column(String(100), nullable=True)
    confidence_score = Column(Float, nullable=True)
    multiple_classifications = Column(Text, nullable=True)  # JSON string for multiple crimes
    detected_objects = Column(Text, nullable=True)  # JSON string
    detected_people_count = Column(Integer, nullable=True)
    
    # Content moderation
    is_inappropriate = Column(Boolean, default=False)
    moderation_flags = Column(Text, nullable=True)  # JSON string
    
    # User tracking
    user_ip = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    def __repr__(self):
        return f'<Video {self.filename}>'

class UserWarning(db.Model):
    __tablename__ = 'user_warnings'
    
    id = Column(Integer, primary_key=True)
    user_ip = Column(String(45), nullable=False)
    warning_count = Column(Integer, default=0)
    last_warning = Column(DateTime, nullable=True)
    is_banned = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<UserWarning {self.user_ip}: {self.warning_count}>'

class ProcessingLog(db.Model):
    __tablename__ = 'processing_logs'
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, nullable=False)
    step = Column(String(100), nullable=False)  # ingestion, duplicate_detection, classification, etc.
    status = Column(String(50), nullable=False)  # started, completed, failed
    message = Column(Text, nullable=True)
    processing_time = Column(Float, nullable=True)  # seconds
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ProcessingLog {self.video_id}: {self.step}>'

class Alert(db.Model):
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, nullable=False)
    alert_type = Column(String(100), nullable=False)  # urgent_crime, people_crowd, etc.
    recipient_type = Column(String(100), nullable=False)  # police, community, etc.
    message = Column(Text, nullable=False)
    sent_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Alert {self.alert_type} for video {self.video_id}>'
