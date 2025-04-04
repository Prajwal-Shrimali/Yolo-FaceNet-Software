import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
from datetime import date, datetime
import os
from deepface import DeepFace
from config import (
    DB_CONFIG, FACE_RECOGNITION, SERVER, FILE_NAMING, DEBUG,
    ensure_directories
)
import mysql.connector
import traceback
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FaceAuthServer")

class FaceAuthServer:
    def __init__(self):
        logger.info("Initializing FaceAuthServer")
        ensure_directories()
        self.db_connection = None
        self.connect_db()

    def connect_db(self):
        """Establish database connection."""
        try:
            logger.debug(f"Connecting to database: {DB_CONFIG['host']}:{DB_CONFIG['database']}")
            self.db_connection = mysql.connector.connect(**DB_CONFIG)
            logger.info("Database connection established")
            
            # Create/update tables if they don't exist
            self._create_tables()
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            traceback.print_exc()
            raise

    def _create_tables(self):
        """Create necessary database tables if they don't exist."""
        try:
            cursor = self.db_connection.cursor()
            
            # Create access_table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_table (
                    access_id INT AUTO_INCREMENT PRIMARY KEY,
                    person_id VARCHAR(20) NOT NULL,
                    patient_id VARCHAR(20) NOT NULL,
                    access_date DATE NOT NULL,
                    visit_type VARCHAR(20) DEFAULT 'regular',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_access (person_id, patient_id, access_date)
                )
            """)
            
            # Check if visit_type column exists, if not add it
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_name = 'access_table' 
                AND column_name = 'visit_type'
            """)
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    ALTER TABLE access_table 
                    ADD COLUMN visit_type VARCHAR(20) DEFAULT 'regular'
                """)
                logger.info("Added visit_type column to access_table")
            
            # Create logging_table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logging_table (
                    log_id INT AUTO_INCREMENT PRIMARY KEY,
                    person_id VARCHAR(20) NOT NULL,
                    patient_id VARCHAR(20) NOT NULL,
                    visit_type VARCHAR(20) NOT NULL,
                    action VARCHAR(20) NOT NULL,
                    reason TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_person ON access_table(person_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_patient ON access_table(patient_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_date ON access_table(access_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_log_person ON logging_table(person_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_log_patient ON logging_table(patient_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_log_timestamp ON logging_table(timestamp)")
            
            self.db_connection.commit()
            logger.info("Database tables and indexes checked/created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            traceback.print_exc()
            self.db_connection.rollback()
        finally:
            cursor.close()

    def base64_to_image(self, base64_string):
        """Convert base64 string to numpy array image."""
        try:
            logger.debug("Converting base64 to image")
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64 string
            img_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode image from base64")
                return None
                
            logger.debug(f"Image decoded successfully: {img.shape}")
            return img
        except Exception as e:
            logger.error(f"Error converting base64 to image: {e}")
            traceback.print_exc()
            return None

    def save_temp_image(self, image):
        """Save image temporarily for DeepFace processing."""
        try:
            temp_path = os.path.join(FACE_RECOGNITION['face_db_path'], FACE_RECOGNITION['temp_image_name'])
            logger.debug(f"Saving temporary image to: {temp_path}")
            cv2.imwrite(temp_path, image)
            return temp_path
        except Exception as e:
            logger.error(f"Error saving temporary image: {e}")
            traceback.print_exc()
            return None

    def verify_face(self, image, role):
        """Verify face using DeepFace find function."""
        try:
            logger.info(f"Verifying {role} face")
            
            # Save image temporarily
            temp_path = self.save_temp_image(image)
            if temp_path is None:
                logger.error("Failed to save temporary image")
                return None
            
            # Determine which folder to search based on role
            folder = FACE_RECOGNITION['doctor_folder'] if role == 'Doctor' else FACE_RECOGNITION['patient_folder']
            search_path = os.path.join(FACE_RECOGNITION['face_db_path'], folder)
            logger.debug(f"Searching in folder: {search_path}")
            
            # Use DeepFace find function
            logger.debug(f"Running DeepFace.find with model: {FACE_RECOGNITION['model_name']}")
            dfs = DeepFace.find(
                img_path=temp_path,
                db_path=search_path,
                model_name=FACE_RECOGNITION['model_name'],
                # distance_metric="cosine",
                enforce_detection=False
            )
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
                logger.debug("Temporary file removed")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")
            
            # Check if we got any results
            if not dfs or len(dfs) == 0 or dfs[0].empty:
                logger.warning(f"No matching face found for {role}")
                return None
            
            # Get the best match
            try:
                best_match = dfs[0].iloc[0]
                # The column name is 'distance' in the current version
                similarity = 1 - best_match['distance']  # Convert distance to similarity
                logger.info(f"Best match similarity: {similarity:.3f}")
                
                if similarity < FACE_RECOGNITION['similarity_threshold']:
                    logger.warning(f"Best match similarity too low: {similarity:.3f}")
                    return None
                
                # Extract person details from filename
                face_path = best_match['identity']
                filename = os.path.basename(face_path)
                logger.debug(f"Matched with file: {filename}")
                
                # Split filename into parts (e.g., DOC_001_John.jpg)
                name_parts = filename.split('.')[0].split('_')  # Remove extension and split
                
                if len(name_parts) >= 3:  # Should have [prefix, number, name]
                    # Extract ID and name
                    prefix = name_parts[0]  # e.g., DOC
                    number = name_parts[1]  # e.g., 001
                    name = name_parts[2]    # e.g., John
                    
                    person_id = f"{prefix}_{number}"  # e.g., DOC_001
                    
                    result = {
                        'person_id': person_id,
                        'name': name,
                        'role': role,
                        'similarity': float(similarity)
                    }
                    logger.info(f"Face verified: {result['name']} ({result['person_id']})")
                    return result
                else:
                    logger.warning(f"Invalid filename format: {filename}. Expected format: PREFIX_NUMBER_NAME.jpg")
                    return None
            except (IndexError, KeyError) as e:
                logger.error(f"Error processing match results: {e}")
                logger.error(f"Available columns: {best_match.index.tolist()}")
                traceback.print_exc()
                return None
            
        except Exception as e:
            logger.error(f"Face verification failed: {e}")
            traceback.print_exc()
            return None

    def check_access(self, person_id: str, patient_id: str, access_date: date):
        """Check if a person has access to a patient on a specific date."""
        try:
            logger.debug(f"Checking access: {person_id} -> {patient_id} on {access_date}")
            cursor = self.db_connection.cursor(dictionary=True)
            query = """
            SELECT * FROM access_table 
            WHERE person_id = %s 
            AND patient_id = %s 
            AND access_date = %s
            """
            cursor.execute(query, (person_id, patient_id, access_date))
            result = cursor.fetchone() is not None
            cursor.close()
            logger.info(f"Access check result: {'Granted' if result else 'Denied'}")
            return result
        except Exception as e:
            logger.error(f"Error checking access: {e}")
            traceback.print_exc()
            return False

    def log_authentication(self, person_id: str, patient_id: str, visit_type: str, action: str, reason: str = None):
        """Log authentication attempts in the system."""
        try:
            logger.debug(f"Logging authentication: {person_id} -> {patient_id}, {action}, {reason}")
            cursor = self.db_connection.cursor()
            query = """
            INSERT INTO logging_table (person_id, patient_id, visit_type, action, reason, timestamp)
            VALUES (%s, %s, %s, %s, %s, NOW())
            """
            cursor.execute(query, (person_id, patient_id, visit_type, action, reason))
            self.db_connection.commit()
            logger.info("Authentication logged successfully")
        except Exception as e:
            logger.error(f"Error logging authentication: {e}")
            traceback.print_exc()
            self.db_connection.rollback()
        finally:
            cursor.close()

    async def handle_websocket(self, websocket):
        """Handle WebSocket connections."""
        logger.info("New WebSocket connection established")
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received message type: {data['type']}")
                    
                    if data['type'] == 'authentication':
                        # Process authentication request
                        logger.debug("Processing authentication request")
                        result = await self.process_authentication(data)
                        logger.debug(f"Sending response: {result}")
                        await websocket.send(json.dumps(result))
                    elif data['type'] == 'registration':
                        # Process registration request
                        logger.debug("Processing registration request")
                        result = await self.process_registration(data)
                        logger.debug(f"Sending response: {result}")
                        await websocket.send(json.dumps(result))
                    elif data['type'] == 'get_doctors':
                        # Get list of doctors
                        logger.debug("Getting doctors list")
                        result = await self.get_doctors_list()
                        logger.debug(f"Sending response: {result}")
                        await websocket.send(json.dumps(result))
                    elif data['type'] == 'get_patients':
                        # Get list of patients
                        logger.debug("Getting patients list")
                        result = await self.get_patients_list()
                        logger.debug(f"Sending response: {result}")
                        await websocket.send(json.dumps(result))
                    elif data['type'] == 'create_access':
                        # Create new access entry
                        logger.debug("Creating access entry")
                        result = await self.create_access_entry(data)
                        logger.debug(f"Sending response: {result}")
                        await websocket.send(json.dumps(result))
                        
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    await websocket.send(json.dumps({
                        'status': 'error',
                        'message': 'Invalid JSON format'
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error handling WebSocket: {e}")
            traceback.print_exc()

    async def process_authentication(self, data):
        """Process authentication request."""
        try:
            logger.info("Processing authentication request")
            
            # Extract data from request
            doctor_image_data = data.get('doctor_image')
            patient_image_data = data.get('patient_image')
            
            if not all([doctor_image_data, patient_image_data]):
                logger.warning("Missing required fields in request")
                return {
                    'status': 'error',
                    'message': 'Missing required fields'
                }
            
            # Convert base64 images to numpy arrays
            logger.debug("Converting doctor image from base64")
            doctor_image = self.base64_to_image(doctor_image_data)
            logger.debug("Converting patient image from base64")
            patient_image = self.base64_to_image(patient_image_data)
            
            if doctor_image is None or patient_image is None:
                logger.error("Failed to convert images from base64")
                return {
                    'status': 'error',
                    'message': 'Invalid image data'
                }
            
            # Save received images for debugging
            try:
                # Create debug directory if it doesn't exist
                debug_dir = os.path.join(FACE_RECOGNITION['face_db_path'], FACE_RECOGNITION['debug_folder'])
                os.makedirs(debug_dir, exist_ok=True)
                
                # Save doctor image
                timestamp = datetime.now().strftime(DEBUG['timestamp_format'])
                doctor_debug_path = os.path.join(debug_dir, f"debug_doctor_{timestamp}.{DEBUG['image_format']}")
                cv2.imwrite(doctor_debug_path, doctor_image)
                logger.info(f"Saved doctor debug image to: {doctor_debug_path}")
                
                # Save patient image
                patient_debug_path = os.path.join(debug_dir, f"debug_patient_{timestamp}.{DEBUG['image_format']}")
                cv2.imwrite(patient_debug_path, patient_image)
                logger.info(f"Saved patient debug image to: {patient_debug_path}")
            except Exception as e:
                logger.error(f"Error saving debug images: {e}")
                traceback.print_exc()
            
            # Verify doctor's face
            logger.info("Verifying doctor's face")
            doctor_result = self.verify_face(doctor_image, 'Doctor')
            if doctor_result is None:
                logger.warning("Doctor verification failed")
                return {
                    'status': 'error',
                    'message': 'Doctor verification failed'
                }
            
            # Verify patient's face
            logger.info("Verifying patient's face")
            patient_result = self.verify_face(patient_image, 'Patient')
            if patient_result is None:
                logger.warning("Patient verification failed")
                return {
                    'status': 'error',
                    'message': 'Patient verification failed'
                }
            
            # Get patient_id from patient verification result
            patient_id = patient_result['person_id']
            logger.debug(f"Patient ID: {patient_id}")
            
            # Check access in database
            access_date = date.today()
            logger.debug(f"Checking access for date: {access_date}")
            has_access = self.check_access(
                doctor_result['person_id'],
                patient_id,
                access_date
            )
            
            # Log authentication attempt
            logger.debug("Logging authentication attempt")
            self.log_authentication(
                doctor_result['person_id'],
                patient_id,
                'face',
                'verify',
                f"Access {'granted' if has_access else 'denied'}"
            )
            
            if has_access:
                logger.info(f"Access granted: Dr. {doctor_result['name']} with {patient_result['name']}")
                return {
                    'status': 'success',
                    'message': f"Access granted: Dr. {doctor_result['name']} with {patient_result['name']}"
                }
            else:
                logger.warning(f"Access denied: Dr. {doctor_result['name']} does not have access to {patient_result['name']}")
                return {
                    'status': 'error',
                    'message': f"Access denied: Dr. {doctor_result['name']} does not have access to {patient_result['name']}"
                }
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e)
            }

    async def process_registration(self, data):
        """Process registration request."""
        try:
            logger.info("Processing registration request")
            
            # Extract data from request
            image_data = data.get('image')
            name = data.get('name')
            role = data.get('role')
            
            if not all([image_data, name, role]):
                logger.warning("Missing required fields in request")
                return {
                    'status': 'error',
                    'message': 'Missing required fields'
                }
            
            # Convert base64 to image
            image = self.base64_to_image(image_data)
            if image is None:
                logger.error("Failed to convert base64 to image")
                return {
                    'status': 'error',
                    'message': 'Invalid image data'
                }
            
            # Determine folder and prefix based on role
            folder = FACE_RECOGNITION['doctor_folder'] if role == 'Doctor' else FACE_RECOGNITION['patient_folder']
            prefix = 'DOC' if role == 'Doctor' else 'PAT'
            
            # Get the next available number
            folder_path = os.path.join(FACE_RECOGNITION['face_db_path'], folder)
            existing_files = [f for f in os.listdir(folder_path) if f.startswith(prefix)]
            if existing_files:
                # Extract numbers and find max
                numbers = [int(f.split('_')[1]) for f in existing_files]
                next_number = max(numbers) + 1
            else:
                next_number = 1
            
            # Format number with leading zeros
            number_str = f"{next_number:03d}"
            
            # Create filename
            filename = f"{prefix}_{number_str}_{name}.jpg"
            file_path = os.path.join(folder_path, filename)
            
            # Save image
            logger.debug(f"Saving image to: {file_path}")
            cv2.imwrite(file_path, image)
            
            # Create person ID
            person_id = f"{prefix}_{number_str}"
            
            # Log the registration
            self.log_authentication(
                person_id=person_id,
                patient_id=None,
                visit_type='registration',
                action='register',
                reason=f"New {role} registered: {name}"
            )
            
            logger.info(f"Registration successful: {person_id}")
            return {
                'status': 'success',
                'message': f'Successfully registered as {role} with ID: {person_id}'
            }
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            traceback.print_exc()
            return {
                'status': 'error',
                'message': f'Registration failed: {str(e)}'
            }

    async def get_doctors_list(self):
        """Get list of all registered doctors."""
        try:
            logger.info("Getting doctors list")
            doctors = []
            
            # Get doctor folder path
            folder_path = os.path.join(FACE_RECOGNITION['face_db_path'], FACE_RECOGNITION['doctor_folder'])
            
            # Get all files in doctor folder
            for filename in os.listdir(folder_path):
                if filename.startswith('DOC_'):
                    # Extract ID and name from filename
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        person_id = f"{parts[0]}_{parts[1]}"
                        name = parts[2].split('.')[0]  # Remove file extension
                        doctors.append({
                            'id': person_id,
                            'name': name
                        })
            
            logger.info(f"Found {len(doctors)} doctors")
            return {
                'status': 'success',
                'doctors': doctors
            }
            
        except Exception as e:
            logger.error(f"Error getting doctors list: {e}")
            traceback.print_exc()
            return {
                'status': 'error',
                'message': f'Failed to get doctors list: {str(e)}'
            }

    async def get_patients_list(self):
        """Get list of all registered patients."""
        try:
            logger.info("Getting patients list")
            patients = []
            
            # Get patient folder path
            folder_path = os.path.join(FACE_RECOGNITION['face_db_path'], FACE_RECOGNITION['patient_folder'])
            
            # Get all files in patient folder
            for filename in os.listdir(folder_path):
                if filename.startswith('PAT_'):
                    # Extract ID and name from filename
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        person_id = f"{parts[0]}_{parts[1]}"
                        name = parts[2].split('.')[0]  # Remove file extension
                        patients.append({
                            'id': person_id,
                            'name': name
                        })
            
            logger.info(f"Found {len(patients)} patients")
            return {
                'status': 'success',
                'patients': patients
            }
            
        except Exception as e:
            logger.error(f"Error getting patients list: {e}")
            traceback.print_exc()
            return {
                'status': 'error',
                'message': f'Failed to get patients list: {str(e)}'
            }

    async def create_access_entry(self, data):
        """Create new access entry in database."""
        try:
            logger.info("Creating access entry")
            
            # Extract data from request
            doctor_id = data.get('doctor_id')
            patient_id = data.get('patient_id')
            access_date = data.get('access_date')
            visit_type = data.get('visit_type', 'regular')
            
            if not all([doctor_id, patient_id, access_date]):
                logger.warning("Missing required fields in request")
                return {
                    'status': 'error',
                    'message': 'Missing required fields'
                }
            
            # Convert date string to date object
            try:
                access_date = datetime.strptime(access_date, '%Y-%m-%d').date()
            except ValueError:
                logger.error("Invalid date format")
                return {
                    'status': 'error',
                    'message': 'Invalid date format. Use YYYY-MM-DD'
                }
            
            # Check if access already exists
            cursor = self.db_connection.cursor(dictionary=True)
            query = """
            SELECT * FROM access_table 
            WHERE person_id = %s 
            AND patient_id = %s 
            AND access_date = %s
            """
            cursor.execute(query, (doctor_id, patient_id, access_date))
            if cursor.fetchone():
                logger.warning("Access entry already exists")
                cursor.close()
                return {
                    'status': 'error',
                    'message': 'Access entry already exists for this date'
                }
            
            # Create new access entry
            query = """
            INSERT INTO access_table (person_id, patient_id, access_date, visit_type)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (doctor_id, patient_id, access_date, visit_type))
            self.db_connection.commit()
            
            # Log the access creation
            self.log_authentication(
                person_id=doctor_id,
                patient_id=patient_id,
                visit_type=visit_type,
                action='create_access',
                reason=f"New access created for {access_date}"
            )
            
            logger.info("Access entry created successfully")
            return {
                'status': 'success',
                'message': 'Access entry created successfully'
            }
            
        except Exception as e:
            logger.error(f"Error creating access entry: {e}")
            traceback.print_exc()
            self.db_connection.rollback()
            return {
                'status': 'error',
                'message': f'Failed to create access entry: {str(e)}'
            }
        finally:
            cursor.close()

    async def start(self):
        """Start the WebSocket server."""
        logger.info(f"Starting server on {SERVER['host']}:{SERVER['port']}")
        server = await websockets.serve(
            self.handle_websocket,
            SERVER['host'],
            SERVER['port']
        )
        logger.info(f"Server started on ws://{SERVER['host']}:{SERVER['port']}")
        await server.wait_closed()

async def main():
    logger.info("Initializing server")
    server = FaceAuthServer()
    await server.start()

if __name__ == "__main__":
    logger.info("Starting Face Authentication Server")
    asyncio.run(main()) 