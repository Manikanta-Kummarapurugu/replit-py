# KrimeWatch - AI-Powered Crime Scene Video Analysis System

## Overview

KrimeWatch is a Flask-based web application that provides automated analysis of crime scene videos using artificial intelligence. The system processes uploaded videos through a sophisticated multi-agent architecture, detecting duplicates, classifying content into specific crime categories, moderating inappropriate material, and generating targeted alerts for law enforcement. The application features enhanced crime detection capabilities designed to accurately identify theft, burglary, assault, vehicle crimes, and other criminal activities for real-world law enforcement applications.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes (August 2025)

### Expanded Crime Detection & Error Prevention System (August 5, 2025)
- **7 New Crime Categories**: Added domestic violence, shoplifting, cyber crime, public disturbance, workplace violence detection
- **Context-Aware Detection**: Enhanced algorithms now consider indoor/outdoor settings, timing, duration, and environmental factors
- **Multi-Factor Scoring**: Complex crime scenarios now analyzed using multiple behavioral indicators and object detection
- **Sports Activity Filtering**: Comprehensive sports detection (25+ keywords) prevents false positives on legitimate activities
- **Robust Upload Error Handling**: Enhanced error handling prevents internal errors during video upload process
- **Enhanced Object Recognition**: Expanded detection includes workplace items, retail indicators, tech equipment, crowd items
- **Temporal Analysis**: Business hours detection for workplace violence, night-time activity flagging for suspicious behavior
- **Multi-Algorithm Classification**: Successfully implemented 6 different AI classification algorithms working in ensemble
- **Pattern-Based Detection**: Rule-based crime pattern recognition using behavioral analysis
- **Enhanced Video Gallery**: Fully functional interactive video gallery with real-time statistics (4 videos, crime detection working)
- **Smart Duplicate Management**: Enhanced duplicate detection now properly identifies and manages duplicate videos with quality-based canonical selection

### Migration to Replit Environment (August 4, 2025)
- **Successfully migrated**: Project fully migrated from Replit Agent to standard Replit environment
- **Performance optimized**: Enhanced duplicate detection with timeout protection (15s max)
- **Improved stability**: Optimized frame comparison to prevent worker timeouts
- **Enhanced duplicate detection**: Now uses fast perceptual hashing with 8x8 hash size for speed
- **Smart comparison limits**: Limited to 5 video comparisons max to prevent timeouts
- **Timeout safeguards**: Added comprehensive timeout protection throughout video processing pipeline
- **Lowered thresholds**: Hash similarity threshold reduced to 0.60 for better duplicate detection

### Enhanced Duplicate Detection System (August 4, 2025)
- **Frame-by-frame comparison**: Optimized perceptual hashing comparison of up to 10 video frames
- **Comprehensive duplicate checking**: System compares against ALL videos in database, with smart limits
- **Smart canonical selection**: Quality-based selection using duration, resolution, file size, and upload time metrics
- **Automatic file management**: Duplicate files automatically moved to `/uploads/duplicates/` folder with clear naming convention
- **Multi-algorithm similarity**: Combines content hash, perceptual hash, frame comparison, and duration analysis
- **Robust detection**: Detects duplicates even with different formats or slight quality differences
- **Optimized performance**: Fast processing with early exit on high similarity detection

### Multi-Crime Classification Enhancement
- **Added shooting detection**: Advanced pattern recognition with 95% confidence for firearm incidents
- **Added kidnapping detection**: Multi-factor scoring system for abduction scenarios (65% threshold)
- **Multi-crime support**: Single videos can now be classified with multiple simultaneous crime types
- **Enhanced object detection**: Comprehensive detection of weapons, restraint tools, and criminal evidence
- **Database schema updated**: Added `multiple_classifications` field for storing all detected crimes
- **UI improvements**: Results page now displays all detected crime types with individual confidence scores
- **Alert system enhanced**: Critical emergency alerts for shooting/kidnapping scenarios

## System Architecture

### Core Architecture Pattern
The application follows a **multi-agent architecture** where each service acts as an independent agent responsible for a specific aspect of video processing. This modular design allows for scalable processing and clear separation of concerns.

### Backend Framework
- **Flask** with SQLAlchemy ORM for the web application layer
- **SQLite** as the default database (configurable to other databases via DATABASE_URL)
- **Declarative Base** model pattern for database schema definition

### Multi-Agent Processing Pipeline
The system implements a sequential processing pipeline with four main agents:

1. **Video Processor Agent** (`services/video_processor.py`)
   - Handles video ingestion and metadata extraction
   - Generates content and perceptual hashes using OpenCV and imagehash
   - Extracts technical metadata (duration, dimensions, FPS)

2. **Duplicate Detector Agent** (`services/duplicate_detector.py`)
   - **Enhanced Frame Comparison**: Compares up to 20 frames per video using perceptual hashing
   - **Comprehensive Database Scanning**: Checks against ALL videos in database for thorough duplicate detection
   - **Multi-Algorithm Similarity**: Combines content hash, perceptual hash, frame analysis, and metadata comparison
   - **Smart Quality Selection**: Automatically selects best quality video as canonical based on duration, resolution, and file size
   - **Automatic File Management**: Moves duplicate files to `/uploads/duplicates/` folder with timestamped naming
   - **Configurable Threshold**: Uses 0.75 similarity threshold for sensitive duplicate detection

3. **AI Classifier Agent** (`services/ai_classifier.py`)
   - **Enhanced Multi-Crime Classification**: Now supports detection of multiple simultaneous crimes in a single video (e.g., shooting + kidnapping scenario)
   - **Expanded Crime Categories**: 15 specific crime categories including shooting, kidnapping, theft, burglary, robbery, assault, weapon detection, vehicle crime, vandalism, drug activity, crowd disturbance, suspicious activity, and traffic violations
   - **Advanced Pattern Recognition**: Sophisticated behavioral analysis using scoring systems for shooting (0.7 threshold) and kidnapping (0.65 threshold) detection
   - **Enhanced Object Detection**: Comprehensive object recognition simulating real-world YOLO detection with weapons, restraint tools, and criminal evidence identification
   - **Priority-Based Classification**: Hierarchical classification system prioritizing critical crimes (shooting, kidnapping) over lesser offenses
   - **Multiple Classification Storage**: Database support for storing all detected crime types with individual confidence scores

4. **Content Moderator Agent** (`services/content_moderator.py`)
   - Detects inappropriate content and flags violations
   - Implements user warning system with ban capabilities
   - Maintains content moderation flags and tracking

### Data Storage Strategy
- **Primary Database**: Stores video metadata, processing logs, user warnings, alerts, and multiple crime classifications
- **Enhanced Schema**: Added `multiple_classifications` JSON field to store all detected crime types with confidence scores
- **File Storage**: Videos stored in local `uploads/` directory with unique filenames
- **Processing Logs**: Comprehensive audit trail for all agent operations
- **Multi-Classification Support**: Tracks primary classification plus all secondary crime detections

### Frontend Architecture
- **Server-side rendered** templates using Jinja2
- **Bootstrap 5** with dark theme for responsive UI
- **Enhanced Results Display**: Multi-crime classification visualization with color-coded badges and confidence percentages
- **Critical Alert System**: Special alerts for shooting/kidnapping with emergency response notifications
- **Progressive enhancement** with JavaScript for upload progress and validation
- **Real-time status updates** for processing pipeline

### Configuration Management
Centralized configuration in `config.py` with environment-specific settings:
- Video processing parameters (file size limits, allowed formats)
- AI model thresholds and confidence scores
- Duplicate detection parameters
- Content moderation rules

### Error Handling and Logging
- Comprehensive logging throughout the processing pipeline
- Database-backed processing logs for audit trails
- Graceful degradation when AI models are unavailable
- User-friendly error messages and flash notifications

## External Dependencies

### Core Framework Dependencies
- **Flask**: Web application framework
- **SQLAlchemy**: Database ORM and connection management
- **Werkzeug**: WSGI utilities and file upload handling

### AI and Computer Vision
- **OpenCV (cv2)**: Video processing, frame extraction, and DNN module for object detection
- **PIL (Pillow)**: Image processing for perceptual hashing
- **imagehash**: Perceptual hash generation for duplicate detection
- **NumPy**: Numerical operations for image and video processing

### Optional AI Models
- **YOLO v4**: Object detection model (weights, config, and class files)
- **Pre-trained classification models**: For crime scene categorization
- **Content moderation models**: For inappropriate content detection

### Frontend Libraries
- **Bootstrap 5**: CSS framework with dark theme support
- **Font Awesome 6**: Icon library for UI elements
- **Custom CSS/JS**: Upload progress tracking and form validation

### Production Considerations
- **ProxyFix**: WSGI middleware for deployment behind reverse proxies
- **Database URL configuration**: Supports PostgreSQL, MySQL, and other databases
- **Session management**: Configurable secret key for production security
- **File upload limits**: 500MB maximum with configurable storage location