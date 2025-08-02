import logging
from datetime import datetime
from app import db
from models import Video, Alert, ProcessingLog
from config import Config

logger = logging.getLogger(__name__)

class AlertGenerator:
    """Agent for generating alerts based on video classification"""
    
    def __init__(self):
        self.name = "AlertGenerator"
        self.urgent_threshold = Config.URGENT_CRIME_CONFIDENCE_THRESHOLD
    
    def generate_alerts(self, video_id):
        """Generate appropriate alerts based on video classification"""
        start_time = datetime.now()
        
        try:
            self._log_processing(video_id, 'alert_generation', 'started',
                               'Starting alert generation')
            
            video = Video.query.get(video_id)
            if not video:
                raise ValueError(f"Video {video_id} not found")
            
            alerts_generated = []
            
            # Generate alerts based on classification
            if video.classification == 'urgent_crime':
                alerts_generated.extend(self._generate_urgent_crime_alerts(video))
            elif video.classification == 'people_crowd':
                alerts_generated.extend(self._generate_people_alerts(video))
            elif video.classification == 'vehicle_traffic':
                alerts_generated.extend(self._generate_traffic_alerts(video))
            elif video.classification == 'property_damage':
                alerts_generated.extend(self._generate_property_alerts(video))
            
            # Log alerts in database
            for alert_data in alerts_generated:
                alert = Alert(
                    video_id=video_id,
                    alert_type=alert_data['type'],
                    recipient_type=alert_data['recipient'],
                    message=alert_data['message']
                )
                db.session.add(alert)
            
            db.session.commit()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            message = f"Generated {len(alerts_generated)} alert(s)"
            self._log_processing(video_id, 'alert_generation', 'completed',
                               message, processing_time)
            
            logger.info(f"Generated {len(alerts_generated)} alerts for video {video_id}")
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._log_processing(video_id, 'alert_generation', 'failed',
                               f'Error: {str(e)}', processing_time)
            logger.error(f"Error generating alerts for video {video_id}: {str(e)}")
            raise
    
    def _generate_urgent_crime_alerts(self, video):
        """Generate alerts for urgent crime incidents"""
        alerts = []
        
        try:
            if video.confidence_score >= self.urgent_threshold:
                # Generate police alert
                police_alert = {
                    'type': 'urgent_crime',
                    'recipient': 'police',
                    'message': self._create_police_alert_message(video)
                }
                alerts.append(police_alert)
                
                # Generate emergency services alert if needed
                if 'violence' in str(video.detected_objects).lower():
                    emergency_alert = {
                        'type': 'emergency_medical',
                        'recipient': 'emergency_services',
                        'message': self._create_emergency_alert_message(video)
                    }
                    alerts.append(emergency_alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating urgent crime alerts: {str(e)}")
            return []
    
    def _generate_people_alerts(self, video):
        """Generate alerts for people/crowd incidents"""
        alerts = []
        
        try:
            if video.detected_people_count and video.detected_people_count > 5:
                # Large crowd detected
                crowd_alert = {
                    'type': 'large_crowd',
                    'recipient': 'community_services',
                    'message': self._create_crowd_alert_message(video)
                }
                alerts.append(crowd_alert)
            
            elif video.detected_people_count > 0:
                # Regular people activity
                people_alert = {
                    'type': 'people_activity',
                    'recipient': 'neighborhood_watch',
                    'message': self._create_people_alert_message(video)
                }
                alerts.append(people_alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating people alerts: {str(e)}")
            return []
    
    def _generate_traffic_alerts(self, video):
        """Generate alerts for traffic/vehicle incidents"""
        alerts = []
        
        try:
            traffic_alert = {
                'type': 'traffic_incident',
                'recipient': 'traffic_control',
                'message': self._create_traffic_alert_message(video)
            }
            alerts.append(traffic_alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating traffic alerts: {str(e)}")
            return []
    
    def _generate_property_alerts(self, video):
        """Generate alerts for property damage incidents"""
        alerts = []
        
        try:
            property_alert = {
                'type': 'property_damage',
                'recipient': 'property_management',
                'message': self._create_property_alert_message(video)
            }
            alerts.append(property_alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating property alerts: {str(e)}")
            return []
    
    def _create_police_alert_message(self, video):
        """Create detailed police alert message"""
        try:
            message = f"""
URGENT CRIME ALERT - KrimeWatch System

Incident Type: {video.classification.replace('_', ' ').title()}
Confidence Level: {video.confidence_score:.2f}
Timestamp: {video.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

Video Details:
- Duration: {video.duration:.1f} seconds
- People Detected: {video.detected_people_count or 'Unknown'}
- Objects Detected: {video.detected_objects or 'Processing...'}

Location: {self._format_location(video)}

Video ID: {video.id}
Immediate Response Recommended: YES

This is an automated alert from the KrimeWatch AI system.
Please verify and respond according to department protocols.
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error creating police alert message: {str(e)}")
            return f"Urgent crime detected - Video ID: {video.id}"
    
    def _create_emergency_alert_message(self, video):
        """Create emergency services alert message"""
        try:
            message = f"""
EMERGENCY MEDICAL ALERT - KrimeWatch System

Potential Medical Emergency Detected
Timestamp: {video.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Location: {self._format_location(video)}

Indicators: Violence or injury may be present
Video ID: {video.id}

This alert was generated automatically. Please coordinate with police response.
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error creating emergency alert message: {str(e)}")
            return f"Emergency situation detected - Video ID: {video.id}"
    
    def _create_crowd_alert_message(self, video):
        """Create crowd alert message"""
        try:
            message = f"""
CROWD ACTIVITY ALERT - KrimeWatch System

Large Crowd Detected: {video.detected_people_count} people
Timestamp: {video.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Location: {self._format_location(video)}

Video ID: {video.id}
Monitoring Recommended: YES

Please assess situation and deploy resources as needed.
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error creating crowd alert message: {str(e)}")
            return f"Large crowd detected - Video ID: {video.id}"
    
    def _create_people_alert_message(self, video):
        """Create general people activity alert message"""
        try:
            message = f"""
PEOPLE ACTIVITY ALERT - KrimeWatch System

People Activity Detected: {video.detected_people_count} person(s)
Timestamp: {video.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Location: {self._format_location(video)}

Video ID: {video.id}
Community Awareness Notice

This is a routine notification for neighborhood watch coordination.
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error creating people alert message: {str(e)}")
            return f"People activity detected - Video ID: {video.id}"
    
    def _create_traffic_alert_message(self, video):
        """Create traffic incident alert message"""
        try:
            message = f"""
TRAFFIC INCIDENT ALERT - KrimeWatch System

Traffic/Vehicle Incident Detected
Timestamp: {video.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Location: {self._format_location(video)}

Video ID: {video.id}
Traffic Management Required

Please assess traffic conditions and implement appropriate measures.
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error creating traffic alert message: {str(e)}")
            return f"Traffic incident detected - Video ID: {video.id}"
    
    def _create_property_alert_message(self, video):
        """Create property damage alert message"""
        try:
            message = f"""
PROPERTY DAMAGE ALERT - KrimeWatch System

Property Damage Incident Detected
Timestamp: {video.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Location: {self._format_location(video)}

Video ID: {video.id}
Property Assessment Required

Please investigate potential property damage or vandalism.
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error creating property alert message: {str(e)}")
            return f"Property damage detected - Video ID: {video.id}"
    
    def _format_location(self, video):
        """Format location information for alerts"""
        try:
            if video.gps_latitude and video.gps_longitude:
                return f"GPS: {video.gps_latitude:.6f}, {video.gps_longitude:.6f}"
            else:
                return "Location: Not specified (IP-based geolocation may be available)"
                
        except Exception as e:
            logger.error(f"Error formatting location: {str(e)}")
            return "Location: Unknown"
    
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
