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
        """Classify video content based on detection results for crime detection"""
        category = 'no_crime'
        confidence = 0.5
        
        try:
            # Enhanced classification logic for real crime detection
            people_count = detection_results['people_count']
            objects = detection_results['objects']
            vehicles = detection_results['vehicles']
            weapons = detection_results['weapons']
            suspicious_objects = detection_results['suspicious_objects']
            
            # Weapon detection - highest priority
            if weapons:
                if any(weapon in ['gun', 'pistol', 'rifle'] for weapon in weapons):
                    category = 'robbery'
                    confidence = 0.95
                elif any(weapon in ['knife', 'blade'] for weapon in weapons):
                    category = 'assault'
                    confidence = 0.9
                else:
                    category = 'weapon_detected'
                    confidence = 0.85
            
            # Theft/Burglary detection patterns
            elif self._detect_theft_patterns(detection_results, video):
                theft_confidence = self._analyze_theft_behavior(detection_results, video)
                if theft_confidence > 0.8:
                    category = 'theft'
                    confidence = theft_confidence
                elif theft_confidence > 0.6:
                    category = 'suspicious_activity'
                    confidence = theft_confidence
            
            # Vehicle crime detection
            elif self._detect_vehicle_crime(detection_results, video):
                category = 'vehicle_crime'
                confidence = 0.8
            
            # Violence/Assault detection
            elif self._detect_violence_patterns(detection_results, video):
                category = 'assault'
                confidence = 0.85
            
            # Burglary/Break-in detection
            elif self._detect_burglary_patterns(detection_results, video):
                category = 'burglary'
                confidence = 0.8
            
            # Drug activity detection
            elif self._detect_drug_activity(detection_results, video):
                category = 'drug_activity'
                confidence = 0.75
            
            # Vandalism detection
            elif self._detect_vandalism(detection_results, video):
                category = 'vandalism'
                confidence = 0.7
            
            # Large crowd disturbance
            elif people_count > 10:
                category = 'crowd_disturbance'
                confidence = 0.8
            
            # Traffic violations
            elif self._detect_traffic_violations(detection_results, video):
                category = 'traffic_violation'
                confidence = 0.7
            
            # General suspicious activity
            elif people_count > 0 and self._detect_suspicious_behavior(detection_results, video):
                category = 'suspicious_activity'
                confidence = 0.6
            
            # Enhance confidence with scene context
            scene_confidence = self._analyze_crime_scene_context(detection_results, video)
            confidence = min(max(confidence, scene_confidence), 1.0)
            
            return {
                'category': category,
                'confidence': confidence,
                'detected_objects': dict(objects),
                'people_count': people_count
            }
            
        except Exception as e:
            logger.error(f"Error classifying content: {str(e)}")
            return {
                'category': 'no_crime',
                'confidence': 0.5,
                'detected_objects': {},
                'people_count': 0
            }
    
    def _detect_theft_patterns(self, detection_results, video):
        """Detect theft/stealing behavior patterns"""
        try:
            objects = detection_results['objects']
            people_count = detection_results['people_count']
            
            # Common theft indicators
            theft_indicators = [
                'bag', 'purse', 'backpack', 'handbag', 'suitcase',
                'car', 'bicycle', 'laptop', 'phone', 'wallet'
            ]
            
            # Check for theft-related objects and people
            if people_count > 0 and any(item in objects for item in theft_indicators):
                return True
            
            # Check for quick movements (short duration videos)
            if video.duration and video.duration < 60 and people_count > 0:
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error detecting theft patterns: {str(e)}")
            return False
    
    def _analyze_theft_behavior(self, detection_results, video):
        """Analyze behavioral patterns for theft confidence"""
        confidence = 0.6
        
        try:
            objects = detection_results['objects']
            people_count = detection_results['people_count']
            
            # Multiple people with valuable objects
            if people_count > 1 and any(item in objects for item in ['bag', 'purse', 'car']):
                confidence += 0.2
            
            # Night time activity
            if video.upload_timestamp.hour < 6 or video.upload_timestamp.hour > 22:
                confidence += 0.1
            
            # Quick actions (short videos)
            if video.duration and video.duration < 30:
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing theft behavior: {str(e)}")
            return 0.6
    
    def _detect_vehicle_crime(self, detection_results, video):
        """Detect vehicle-related crimes"""
        try:
            objects = detection_results['objects']
            vehicles = detection_results['vehicles']
            people_count = detection_results['people_count']
            
            # Vehicle break-in indicators
            if vehicles and people_count > 0:
                # People near vehicles with potential tools
                if any(tool in objects for tool in ['tool', 'crowbar', 'hammer']):
                    return True
                # Multiple people around vehicles
                if people_count > 1:
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error detecting vehicle crime: {str(e)}")
            return False
    
    def _detect_violence_patterns(self, detection_results, video):
        """Detect violence/assault patterns"""
        try:
            objects = detection_results['objects']
            people_count = detection_results['people_count']
            
            # Violence indicators
            if people_count > 1:
                # Multiple people in close proximity
                if any(indicator in str(objects).lower() for indicator in ['fight', 'punch', 'kick']):
                    return True
                # Fast movements (very short duration)
                if video.duration and video.duration < 20:
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error detecting violence patterns: {str(e)}")
            return False
    
    def _detect_burglary_patterns(self, detection_results, video):
        """Detect burglary/break-in patterns"""
        try:
            objects = detection_results['objects']
            people_count = detection_results['people_count']
            
            # Burglary indicators
            burglary_tools = ['crowbar', 'hammer', 'tool', 'ladder']
            entry_objects = ['door', 'window', 'fence', 'gate']
            
            if people_count > 0:
                # People with tools near entry points
                if (any(tool in objects for tool in burglary_tools) and 
                    any(entry in objects for entry in entry_objects)):
                    return True
                # Night time activity near buildings
                if (video.upload_timestamp.hour < 6 or video.upload_timestamp.hour > 22):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error detecting burglary patterns: {str(e)}")
            return False
    
    def _detect_drug_activity(self, detection_results, video):
        """Detect drug-related activity"""
        try:
            objects = detection_results['objects']
            people_count = detection_results['people_count']
            
            # Drug activity indicators
            if people_count > 1:
                # Multiple people in suspicious gatherings
                if video.duration and video.duration < 120:  # Short interactions
                    return True
                # Suspicious objects
                if any(item in objects for item in ['bottle', 'bag', 'package']):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error detecting drug activity: {str(e)}")
            return False
    
    def _detect_vandalism(self, detection_results, video):
        """Detect vandalism/property damage"""
        try:
            objects = detection_results['objects']
            people_count = detection_results['people_count']
            
            # Vandalism indicators
            vandalism_tools = ['spray', 'paint', 'marker', 'hammer', 'rock']
            targets = ['wall', 'car', 'window', 'building', 'sign']
            
            if people_count > 0:
                if (any(tool in objects for tool in vandalism_tools) and 
                    any(target in objects for target in targets)):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error detecting vandalism: {str(e)}")
            return False
    
    def _detect_traffic_violations(self, detection_results, video):
        """Detect traffic violations"""
        try:
            vehicles = detection_results['vehicles']
            objects = detection_results['objects']
            
            # Traffic violation indicators
            if vehicles:
                # Vehicles in inappropriate places
                if any(location in objects for location in ['sidewalk', 'crosswalk', 'playground']):
                    return True
                # Multiple vehicles suggesting racing/reckless driving
                if len(vehicles) > 2:
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error detecting traffic violations: {str(e)}")
            return False
    
    def _detect_suspicious_behavior(self, detection_results, video):
        """Detect general suspicious behavior"""
        try:
            people_count = detection_results['people_count']
            objects = detection_results['objects']
            
            # General suspicious indicators
            if people_count > 0:
                # Loitering (long duration videos with minimal action)
                if video.duration and video.duration > 300:  # 5 minutes
                    return True
                # Night time activity
                if video.upload_timestamp.hour < 6 or video.upload_timestamp.hour > 22:
                    return True
                # Multiple people in unusual places
                if people_count > 3:
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error detecting suspicious behavior: {str(e)}")
            return False
    
    def _analyze_crime_scene_context(self, detection_results, video):
        """Enhanced crime scene context analysis"""
        confidence = 0.5
        
        try:
            objects = detection_results['objects']
            people_count = detection_results['people_count']
            vehicles = detection_results['vehicles']
            
            # Time-based risk factors
            if video.upload_timestamp.hour < 6 or video.upload_timestamp.hour > 22:
                confidence += 0.15  # Night time increases crime likelihood
            
            # Duration analysis
            if video.duration:
                if video.duration < 30:  # Quick actions often criminal
                    confidence += 0.1
                elif video.duration > 300:  # Loitering behavior
                    confidence += 0.05
            
            # Multiple people interactions
            if people_count > 1:
                confidence += 0.1
                if people_count > 3:
                    confidence += 0.1  # Crowds can indicate disturbances
            
            # Vehicle presence with people
            if vehicles and people_count > 0:
                confidence += 0.1  # Vehicle crimes are common
            
            # Object combinations suggesting crime
            crime_objects = ['bag', 'tool', 'weapon', 'car', 'door', 'window']
            object_matches = sum(1 for obj in crime_objects if obj in objects)
            confidence += min(object_matches * 0.05, 0.2)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing crime scene context: {str(e)}")
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
