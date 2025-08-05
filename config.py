import os

class Config:
    # Video processing settings
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    
    # Duplicate detection settings - Enhanced for better detection
    DUPLICATE_TIME_WINDOW = 2 * 60 * 60  # 2 hours in seconds
    DUPLICATE_DISTANCE_RADIUS = 1609.34  # 1 mile in meters
    HASH_SIMILARITY_THRESHOLD = 0.60  # Lowered to 0.60 for more aggressive duplicate detection
    FRAME_SIMILARITY_THRESHOLD = 0.70  # Specific threshold for frame-by-frame comparison
    
    # Classification confidence thresholds
    CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.7
    URGENT_CRIME_CONFIDENCE_THRESHOLD = 0.8
    
    # Warning system
    MAX_WARNINGS = 3
    
    # Crime scene categories - Expanded for comprehensive detection
    CRIME_CATEGORIES = {
        'shooting': 'Shooting/Firearms',
        'kidnapping': 'Kidnapping/Abduction',
        'domestic_violence': 'Domestic Violence',
        'shoplifting': 'Shoplifting/Retail Theft',
        'cyber_crime': 'Cyber Crime/Fraud',
        'public_disturbance': 'Public Disturbance/Riot',
        'workplace_violence': 'Workplace Violence',
        'theft': 'Theft/Stealing',
        'burglary': 'Burglary/Break-in',
        'vehicle_crime': 'Vehicle Crime',
        'assault': 'Assault/Violence',
        'robbery': 'Armed Robbery',
        'vandalism': 'Vandalism/Property Damage',
        'suspicious_activity': 'Suspicious Activity',
        'drug_activity': 'Drug Related Activity',
        'weapon_detected': 'Weapon Present',
        'crowd_disturbance': 'Crowd/Disturbance',
        'traffic_violation': 'Traffic Violation',
        'other_crime': 'Other Criminal Activity',
        'no_crime': 'No Crime Detected'
    }
    
    # Content moderation flags
    INAPPROPRIATE_CONTENT_TYPES = [
        'explicit_content',
        'violence_graphic',
        'harassment',
        'hate_speech',
        'spam_irrelevant'
    ]
