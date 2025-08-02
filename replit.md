# KrimeWatch - AI-Powered Crime Scene Video Analysis System

## Overview

KrimeWatch is a Flask-based web application that provides automated analysis of crime scene videos using artificial intelligence. The system processes uploaded videos through a multi-agent architecture, detecting duplicates, classifying content, moderating inappropriate material, and generating alerts for law enforcement. The application is designed to help streamline the analysis of crime scene footage by automating the initial triage and classification process.

## User Preferences

Preferred communication style: Simple, everyday language.

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
   - Identifies duplicate videos using spatial and temporal proximity
   - Uses configurable time windows (2 hours) and distance radius (1 mile)
   - Implements hash-based similarity comparison with 0.85 threshold

3. **AI Classifier Agent** (`services/ai_classifier.py`)
   - Classifies videos into predefined crime categories
   - Uses OpenCV DNN module with YOLO for object detection
   - Implements confidence thresholds for classification accuracy

4. **Content Moderator Agent** (`services/content_moderator.py`)
   - Detects inappropriate content and flags violations
   - Implements user warning system with ban capabilities
   - Maintains content moderation flags and tracking

### Data Storage Strategy
- **Primary Database**: Stores video metadata, processing logs, user warnings, and alerts
- **File Storage**: Videos stored in local `uploads/` directory with unique filenames
- **Processing Logs**: Comprehensive audit trail for all agent operations

### Frontend Architecture
- **Server-side rendered** templates using Jinja2
- **Bootstrap 5** with dark theme for responsive UI
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