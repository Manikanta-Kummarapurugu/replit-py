import os
import cv2
import json
import logging
import numpy as np
from datetime import datetime
from app import db
from models import Video, ProcessingLog
from config import Config

logger = logging.getLogger(__name__)

class AIClassifier:
    """Agent for AI-powered video classification"""
    
    def __init__(self):
        self.name = "AIClassifier"
        self.confidence_threshold = Config.CLASSIFICATION_CONFIDENCE_THRESHOLD
        self.categories = Config.CRIME_CATEGORIES
        
        # Initialize pre-trained models (using OpenCV DNN for object detection)
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained AI models"""
        try:
            # For demonstration, we'll use OpenCV's DNN module with pre-trained models
            # In production, you would load specialized crime scene classification models
            
            # Load YOLO for object detection
            self.yolo_net = None
            self.yolo_classes = []
            
            # Try to load YOLO model if available
            try:
                # These would be downloaded/provided in a real deployment
                weights_path = "models/yolov4.weights"
                config_path = "models/yolov4.cfg"
                classes_path = "models/coco.names"
                
                if all(os.path.exists(p) for p in [weights_path, config_path, classes_path]):
                    self.yolo_net = cv2.dnn.readNet(weights_path, config_path)
                    with open(classes_path, 'r') as f:
                        self.yolo_classes = [line.strip() for line in f.readlines()]
                    logger.info("YOLO model loaded successfully")
                else:
                    logger.warning("YOLO model files not found, using basic classification")
            except Exception as e:
                logger.warning(f"Could not load YOLO model: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error loading AI models: {str(e)}")
    
    def classify_video(self, video_id):
        """Classify video content into crime scene categories"""
        start_time = datetime.now()
        
        try:
            self._log_processing(video_id, 'classification', 'started',
                               'Starting AI classification')
            
            video = Video.query.get(video_id)
            if not video:
                raise ValueError(f"Video {video_id} not found")
            
            # Extract frames for analysis
            frames = self._extract_sample_frames(video.file_path)
            
            if not frames:
                raise ValueError("Could not extract frames from video")
            
            # Perform object detection and people counting
            detection_results = self._detect_objects_and_people(frames)
            
            # Classify video content based on detected objects and scenes
            classification_result = self._classify_content(detection_results, video)
            
            # Update video record with classification results
            video.classification = classification_result['category']
            video.confidence_score = classification_result['confidence']
            video.detected_objects = json.dumps(classification_result['detected_objects'])
            video.detected_people_count = classification_result['people_count']
            
            db.session.commit()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            message = f"Classified as: {classification_result['category']} (confidence: {classification_result['confidence']:.2f})"
            self._log_processing(video_id, 'classification', 'completed',
                               message, processing_time)
            
            logger.info(f"Video {video_id} classified successfully")
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._log_processing(video_id, 'classification', 'failed',
                               f'Error: {str(e)}', processing_time)
            logger.error(f"Error classifying video {video_id}: {str(e)}")
            raise
    
    def _extract_sample_frames(self, file_path, max_frames=20):
        """Extract sample frames from video for analysis"""
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
    
    def _detect_objects_and_people(self, frames):
        """Detect objects and people in video frames"""
        results = {
            'objects': {},
            'people_count': 0,
            'vehicles': [],
            'weapons': [],
            'suspicious_objects': []
        }
        
        try:
            for frame in frames:
                frame_results = self._analyze_frame(frame)
                
                # Aggregate object counts
                for obj, count in frame_results.get('objects', {}).items():
                    results['objects'][obj] = results['objects'].get(obj, 0) + count
                
                # Update maximum people count
                results['people_count'] = max(results['people_count'], 
                                            frame_results.get('people_count', 0))
                
                # Collect vehicles and weapons
                results['vehicles'].extend(frame_results.get('vehicles', []))
                results['weapons'].extend(frame_results.get('weapons', []))
                results['suspicious_objects'].extend(frame_results.get('suspicious_objects', []))
            
            return results
            
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            return results
    
    def _analyze_frame(self, frame):
        """Analyze a single frame for objects and people"""
        results = {
            'objects': {},
            'people_count': 0,
            'vehicles': [],
            'weapons': [],
            'suspicious_objects': []
        }
        
        try:
            if self.yolo_net is not None:
                # Use YOLO for object detection
                results = self._yolo_detection(frame)
            else:
                # Use basic OpenCV detection methods
                results = self._basic_detection(frame)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {str(e)}")
            return results
    
    def _yolo_detection(self, frame):
        """Perform YOLO-based object detection"""
        results = {
            'objects': {},
            'people_count': 0,
            'vehicles': [],
            'weapons': [],
            'suspicious_objects': []
        }
        
        try:
            height, width = frame.shape[:2]
            
            # Create blob from frame
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.yolo_net.setInput(blob)
            
            # Run inference
            layer_names = self.yolo_net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
            outputs = self.yolo_net.forward(output_layers)
            
            # Process detections
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5:
                        class_name = self.yolo_classes[class_id] if class_id < len(self.yolo_classes) else "unknown"
                        
                        # Count objects
                        results['objects'][class_name] = results['objects'].get(class_name, 0) + 1
                        
                        # Special handling for people, vehicles, weapons
                        if class_name == 'person':
                            results['people_count'] += 1
                        elif class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                            results['vehicles'].append(class_name)
                        elif class_name in ['knife', 'gun']:  # Add more weapon classes as needed
                            results['weapons'].append(class_name)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {str(e)}")
            return self._basic_detection(frame)
    
    def _basic_detection(self, frame):
        """Basic detection using OpenCV methods (fallback)"""
        results = {
            'objects': {},
            'people_count': 0,
            'vehicles': [],
            'weapons': [],
            'suspicious_objects': []
        }
        
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use Haar cascades for people detection (basic)
            people_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect people
            people = people_cascade.detectMultiScale(gray, 1.1, 4)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            results['people_count'] = max(len(people), len(faces))
            results['objects']['person'] = results['people_count']
            
            # Basic motion detection could be added here
            # Vehicle detection using basic methods could be added
            
            return results
            
        except Exception as e:
            logger.error(f"Error in basic detection: {str(e)}")
            return results
    
    def _classify_content(self, detection_results, video):
        """Classify video content based on detection results"""
        category = 'other_informational'
        confidence = 0.5
        
        try:
            # Classification logic based on detected objects and context
            people_count = detection_results['people_count']
            objects = detection_results['objects']
            vehicles = detection_results['vehicles']
            weapons = detection_results['weapons']
            
            # High-priority classification: Urgent Crime
            if weapons or 'violence' in str(objects).lower():
                category = 'urgent_crime'
                confidence = 0.9
            
            # People/Crowd classification
            elif people_count > 5:
                category = 'people_crowd'
                confidence = 0.8
            elif people_count > 0:
                category = 'people_crowd'
                confidence = 0.7
            
            # Vehicle/Traffic classification
            elif vehicles or any(v in objects for v in ['car', 'truck', 'bus', 'motorcycle']):
                category = 'vehicle_traffic'
                confidence = 0.75
            
            # Property damage classification (basic heuristics)
            elif any(keyword in str(objects).lower() for keyword in ['damage', 'break', 'fire']):
                category = 'property_damage'
                confidence = 0.7
            
            # Scene analysis for better classification
            scene_confidence = self._analyze_scene_context(detection_results, video)
            confidence = max(confidence, scene_confidence)
            
            return {
                'category': category,
                'confidence': confidence,
                'detected_objects': dict(objects),
                'people_count': people_count
            }
            
        except Exception as e:
            logger.error(f"Error classifying content: {str(e)}")
            return {
                'category': 'other_informational',
                'confidence': 0.5,
                'detected_objects': {},
                'people_count': 0
            }
    
    def _analyze_scene_context(self, detection_results, video):
        """Analyze scene context for better classification"""
        confidence = 0.5
        
        try:
            # Time-based analysis (night scenes might be more suspicious)
            if video.upload_timestamp.hour < 6 or video.upload_timestamp.hour > 22:
                confidence += 0.1
            
            # Duration analysis (very short videos might be urgent)
            if video.duration and video.duration < 30:
                confidence += 0.1
            
            # Object combination analysis
            objects = detection_results['objects']
            if 'person' in objects and len(detection_results['vehicles']) > 0:
                confidence += 0.1  # People + vehicles might indicate traffic incident
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing scene context: {str(e)}")
            return 0.5
    
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
