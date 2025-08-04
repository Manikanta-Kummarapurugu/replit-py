import os
import uuid
import logging
from datetime import datetime
from flask import render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from app import app, db
from models import Video, UserWarning, ProcessingLog
from services.video_processor import VideoProcessor
from services.duplicate_detector import DuplicateDetector
from services.ai_classifier import AIClassifier
from services.content_moderator import ContentModerator
from services.alert_generator import AlertGenerator
from config import Config

logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def get_client_ip():
    """Get the real client IP address"""
    if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
        return request.environ['REMOTE_ADDR']
    else:
        return request.environ['HTTP_X_FORWARDED_FOR']

def check_user_warnings(ip_address):
    """Check if user has exceeded warning limit"""
    warning = UserWarning.query.filter_by(user_ip=ip_address).first()
    if warning and warning.is_banned:
        return False, "Your account has been banned due to repeated violations."
    return True, None

@app.route('/')
def index():
    """Main upload page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing"""
    try:
        # Check if user is banned
        client_ip = get_client_ip()
        is_allowed, message = check_user_warnings(client_ip)
        if not is_allowed:
            if message:
                flash(message, 'error')
            return redirect(url_for('index'))
        
        # Check if file was uploaded
        if 'video' not in request.files:
            flash('No video file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['video']
        if file.filename == '':
            flash('No video file selected', 'error')
            return redirect(url_for('index'))
        
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a video file.', 'error')
            return redirect(url_for('index'))
        
        # Generate unique filename
        if file.filename:
            filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(filepath)
            file_size = os.path.getsize(filepath)
            
            # Create video record
            video = Video(
                filename=filename,
                original_filename=secure_filename(file.filename),
                file_path=filepath,
                file_size=file_size,
                user_ip=client_ip,
                user_agent=request.headers.get('User-Agent', '')[:500]
            )
            
            db.session.add(video)
            db.session.commit()
            
            # Start processing pipeline
            process_video_pipeline(video.id)
            
            return redirect(url_for('upload_result', video_id=video.id))
        else:
            flash('Invalid filename', 'error')
            return redirect(url_for('index'))
        
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        flash('An error occurred while uploading the video', 'error')
        return redirect(url_for('index'))

def process_video_pipeline(video_id):
    """Process video through the AI pipeline"""
    try:
        # Step 1: Video ingestion and metadata extraction
        processor = VideoProcessor()
        processor.process_video(video_id)
        
        # Step 2: Content moderation
        moderator = ContentModerator()
        moderation_result = moderator.moderate_video(video_id)
        
        if moderation_result['is_inappropriate']:
            handle_inappropriate_content(video_id)
            return
        
        # Step 3: Duplicate detection
        detector = DuplicateDetector()
        detector.detect_duplicates(video_id)
        
        # Step 4: AI classification (only for non-duplicates or canonical videos) - with timeout protection
        video = Video.query.get(video_id)
        if video and (not video.is_duplicate or video.canonical_video_id is None):
            try:
                classifier = AIClassifier()
                
                # Set processing timeout to prevent worker timeout
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("Classification timeout")
                
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(20)  # 20 second timeout
                
                try:
                    classifier.classify_video(video_id)
                except TimeoutError:
                    logger.warning(f"Classification timed out for video {video_id}, using fallback")
                    # Quick fallback classification
                    video.category = 'other'
                    video.confidence = 0.5
                    video.classification = 'timeout_fallback'
                    db.session.commit()
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                
                # Step 5: Alert generation (quick)
                alert_generator = AlertGenerator()
                alert_generator.generate_alerts(video_id)
                
            except Exception as e:
                logger.error(f"Error in classification step: {str(e)}")
                # Set basic classification and continue
                video.category = 'other'
                video.confidence = 0.5
                video.classification = 'error_fallback'
                db.session.commit()
        
        # Update status
        if video:
            video.status = 'completed'
            db.session.commit()
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        video = Video.query.get(video_id)
        if video:
            video.status = 'failed'
            db.session.commit()

def handle_inappropriate_content(video_id):
    """Handle inappropriate content detection"""
    video = Video.query.get(video_id)
    if not video:
        return
    
    # Update warning count for user
    warning = UserWarning.query.filter_by(user_ip=video.user_ip).first()
    if not warning:
        warning = UserWarning()
        warning.user_ip = video.user_ip
        warning.warning_count = 1
        db.session.add(warning)
    else:
        warning.warning_count += 1
        warning.last_warning = datetime.now()
    
    # Ban user if they exceed warning limit
    if warning.warning_count >= Config.MAX_WARNINGS:
        warning.is_banned = True
    
    video.status = 'inappropriate'
    db.session.commit()

@app.route('/result/<int:video_id>')
def upload_result(video_id):
    """Display processing results"""
    video = Video.query.get_or_404(video_id)
    
    # Get processing logs
    logs = ProcessingLog.query.filter_by(video_id=video_id).order_by(ProcessingLog.timestamp).all()
    
    # Get canonical video if this is a duplicate
    canonical_video = None
    if video.is_duplicate and video.canonical_video_id:
        canonical_video = Video.query.get(video.canonical_video_id)
    
    return render_template('upload_result.html', 
                         video=video, 
                         logs=logs, 
                         canonical_video=canonical_video)

@app.route('/api/video/<int:video_id>/status')
def video_status(video_id):
    """API endpoint to check video processing status"""
    video = Video.query.get_or_404(video_id)
    
    # Parse multiple classifications if available
    multiple_classifications = []
    if video.multiple_classifications:
        try:
            import json
            multiple_classifications = json.loads(video.multiple_classifications)
        except:
            multiple_classifications = []
    
    return jsonify({
        'id': video.id,
        'status': video.status,
        'classification': video.classification,
        'confidence_score': video.confidence_score,
        'multiple_classifications': multiple_classifications,
        'is_duplicate': video.is_duplicate,
        'detected_people_count': video.detected_people_count,
        'is_inappropriate': video.is_inappropriate
    })

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 500MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))
