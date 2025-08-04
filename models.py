from app import db
from datetime import datetime

class Video(db.Model):
    __tablename__ = 'videos'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    duration = db.Column(db.Float, nullable=True)
    upload_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Metadata
    video_hash = db.Column(db.String(128), nullable=True)
    perceptual_hash = db.Column(db.String(128), nullable=True)
    gps_latitude = db.Column(db.Float, nullable=True)
    gps_longitude = db.Column(db.Float, nullable=True)
    width = db.Column(db.Integer, nullable=True)
    height = db.Column(db.Integer, nullable=True)
    fps = db.Column(db.Float, nullable=True)
    
    # Processing status
    status = db.Column(db.String(50), default='processing')  # processing, completed, failed
    is_duplicate = db.Column(db.Boolean, default=False)
    canonical_video_id = db.Column(db.Integer, nullable=True)
    duplicate_group_id = db.Column(db.String(128), nullable=True)
    
    # Classification results
    classification = db.Column(db.String(100), nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    multiple_classifications = db.Column(db.Text, nullable=True)  # JSON string for multiple crimes
    detected_objects = db.Column(db.Text, nullable=True)  # JSON string
    detected_people_count = db.Column(db.Integer, nullable=True)
    
    # Content moderation
    is_inappropriate = db.Column(db.Boolean, default=False)
    moderation_flags = db.Column(db.Text, nullable=True)  # JSON string
    
    # User tracking
    user_ip = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(500), nullable=True)
    
    def __repr__(self):
        return f'<Video {self.filename}>'

class UserWarning(db.Model):
    __tablename__ = 'user_warnings'
    
    id = db.Column(db.Integer, primary_key=True)
    user_ip = db.Column(db.String(45), nullable=False)
    warning_count = db.Column(db.Integer, default=0)
    last_warning = db.Column(db.DateTime, nullable=True)
    is_banned = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<UserWarning {self.user_ip}: {self.warning_count}>'

class ProcessingLog(db.Model):
    __tablename__ = 'processing_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, nullable=False)
    step = db.Column(db.String(100), nullable=False)  # ingestion, duplicate_detection, classification, etc.
    status = db.Column(db.String(50), nullable=False)  # started, completed, failed
    message = db.Column(db.Text, nullable=True)
    processing_time = db.Column(db.Float, nullable=True)  # seconds
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ProcessingLog {self.video_id}: {self.step}>'

class Alert(db.Model):
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, nullable=False)
    alert_type = db.Column(db.String(100), nullable=False)  # urgent_crime, people_crowd, etc.
    recipient_type = db.Column(db.String(100), nullable=False)  # police, community, etc.
    message = db.Column(db.Text, nullable=False)
    sent_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Alert {self.alert_type} for video {self.video_id}>'
