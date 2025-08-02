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
            if video.classification in ['theft', 'burglary', 'robbery', 'assault', 'weapon_detected']:
                alerts_generated.extend(self._generate_urgent_crime_alerts(video))
            elif video.classification == 'vehicle_crime':
                alerts_generated.extend(self._generate_vehicle_crime_alerts(video))
            elif video.classification == 'vandalism':
                alerts_generated.extend(self._generate_vandalism_alerts(video))
            elif video.classification == 'drug_activity':
                alerts_generated.extend(self._generate_drug_alerts(video))
            elif video.classification == 'crowd_disturbance':
                alerts_generated.extend(self._generate_crowd_alerts(video))
            elif video.classification == 'suspicious_activity':
                alerts_generated.extend(self._generate_suspicious_alerts(video))
            elif video.classification == 'traffic_violation':
                alerts_generated.extend(self._generate_traffic_alerts(video))
            
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
                    'type': video.classification,
                    'recipient': 'police',
                    'message': self._create_police_alert_message(video)
                }
                alerts.append(police_alert)
                
                # Generate emergency services alert for violent crimes
                if video.classification in ['assault', 'robbery', 'weapon_detected']:
                    emergency_alert = {
                        'type': 'emergency_medical',
                        'recipient': 'emergency_services',
                        'message': self._create_emergency_alert_message(video)
                    }
                    alerts.append(emergency_alert)
                    
                # Generate detective unit alert for theft/burglary
                if video.classification in ['theft', 'burglary']:
                    detective_alert = {
                        'type': 'detective_investigation',
                        'recipient': 'detective_unit',
                        'message': self._create_detective_alert_message(video)
                    }
                    alerts.append(detective_alert)
            
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
    
    def _generate_vehicle_crime_alerts(self, video):
        """Generate alerts for vehicle crime incidents"""
        alerts = []
        
        try:
            vehicle_alert = {
                'type': 'vehicle_crime',
                'recipient': 'police',
                'message': self._create_vehicle_crime_alert_message(video)
            }
            alerts.append(vehicle_alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating vehicle crime alerts: {str(e)}")
            return []
    
    def _generate_vandalism_alerts(self, video):
        """Generate alerts for vandalism incidents"""
        alerts = []
        
        try:
            vandalism_alert = {
                'type': 'vandalism',
                'recipient': 'property_management',
                'message': self._create_vandalism_alert_message(video)
            }
            alerts.append(vandalism_alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating vandalism alerts: {str(e)}")
            return []
    
    def _generate_drug_alerts(self, video):
        """Generate alerts for drug activity"""
        alerts = []
        
        try:
            drug_alert = {
                'type': 'drug_activity',
                'recipient': 'narcotics_unit',
                'message': self._create_drug_alert_message(video)
            }
            alerts.append(drug_alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating drug alerts: {str(e)}")
            return []
    
    def _generate_crowd_alerts(self, video):
        """Generate alerts for crowd disturbances"""
        alerts = []
        
        try:
            crowd_alert = {
                'type': 'crowd_disturbance',
                'recipient': 'riot_control',
                'message': self._create_crowd_disturbance_alert_message(video)
            }
            alerts.append(crowd_alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating crowd alerts: {str(e)}")
            return []
    
    def _generate_suspicious_alerts(self, video):
        """Generate alerts for suspicious activity"""
        alerts = []
        
        try:
            suspicious_alert = {
                'type': 'suspicious_activity',
                'recipient': 'patrol_unit',
                'message': self._create_suspicious_alert_message(video)
            }
            alerts.append(suspicious_alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating suspicious alerts: {str(e)}")
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
    
    def _create_detective_alert_message(self, video):
        """Create detective unit alert message"""
        try:
            message = f"""
DETECTIVE INVESTIGATION ALERT - KrimeWatch System

{video.classification.replace('_', ' ').title()} Case Detected
Confidence Level: {video.confidence_score:.2f}
Timestamp: {video.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Location: {self._format_location(video)}

Case Details:
- Crime Type: {video.classification.replace('_', ' ').title()}
- People Involved: {video.detected_people_count or 'Unknown'}
- Evidence Objects: {video.detected_objects or 'Processing...'}

Video ID: {video.id}
Investigation Priority: HIGH

This case requires detective investigation and evidence collection.
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error creating detective alert message: {str(e)}")
            return f"Detective investigation required - Video ID: {video.id}"
    
    def _create_vehicle_crime_alert_message(self, video):
        """Create vehicle crime alert message"""
        try:
            message = f"""
VEHICLE CRIME ALERT - KrimeWatch System

Vehicle Crime Detected
Confidence Level: {video.confidence_score:.2f}
Timestamp: {video.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Location: {self._format_location(video)}

Incident Details:
- People Detected: {video.detected_people_count or 'Unknown'}
- Objects: {video.detected_objects or 'Processing...'}

Video ID: {video.id}
Response Required: IMMEDIATE

Possible vehicle break-in, theft, or vandalism detected.
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error creating vehicle crime alert message: {str(e)}")
            return f"Vehicle crime detected - Video ID: {video.id}"
    
    def _create_vandalism_alert_message(self, video):
        """Create vandalism alert message"""
        try:
            message = f"""
VANDALISM ALERT - KrimeWatch System

Property Vandalism Detected
Confidence Level: {video.confidence_score:.2f}
Timestamp: {video.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Location: {self._format_location(video)}

Incident Details:
- People Involved: {video.detected_people_count or 'Unknown'}
- Damage Assessment Needed

Video ID: {video.id}
Property Management Required

Please assess and document property damage for insurance claims.
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error creating vandalism alert message: {str(e)}")
            return f"Vandalism detected - Video ID: {video.id}"
    
    def _create_drug_alert_message(self, video):
        """Create drug activity alert message"""
        try:
            message = f"""
DRUG ACTIVITY ALERT - KrimeWatch System

Suspected Drug Activity Detected
Confidence Level: {video.confidence_score:.2f}
Timestamp: {video.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Location: {self._format_location(video)}

Activity Details:
- People Involved: {video.detected_people_count or 'Unknown'}
- Suspicious Behavior Patterns Detected

Video ID: {video.id}
Narcotics Unit Response Required

Surveillance and investigation recommended.
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error creating drug alert message: {str(e)}")
            return f"Drug activity detected - Video ID: {video.id}"
    
    def _create_crowd_disturbance_alert_message(self, video):
        """Create crowd disturbance alert message"""
        try:
            message = f"""
CROWD DISTURBANCE ALERT - KrimeWatch System

Large Crowd Disturbance Detected
Confidence Level: {video.confidence_score:.2f}
Timestamp: {video.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Location: {self._format_location(video)}

Crowd Details:
- Estimated People: {video.detected_people_count or 'Multiple'}
- Disturbance Level: Requires Attention

Video ID: {video.id}
Riot Control Assessment Required

Monitor situation and deploy crowd control resources as needed.
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error creating crowd disturbance alert message: {str(e)}")
            return f"Crowd disturbance detected - Video ID: {video.id}"
    
    def _create_suspicious_alert_message(self, video):
        """Create suspicious activity alert message"""
        try:
            message = f"""
SUSPICIOUS ACTIVITY ALERT - KrimeWatch System

Suspicious Behavior Detected
Confidence Level: {video.confidence_score:.2f}
Timestamp: {video.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Location: {self._format_location(video)}

Activity Details:
- People Involved: {video.detected_people_count or 'Unknown'}
- Behavior Pattern: Requires Investigation

Video ID: {video.id}
Patrol Unit Response Recommended

Increased surveillance and patrol presence advised.
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error creating suspicious alert message: {str(e)}")
            return f"Suspicious activity detected - Video ID: {video.id}"
    
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
