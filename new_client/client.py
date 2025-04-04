import cv2
import numpy as np
import websockets
import asyncio
import json
import base64
from ultralytics import YOLO
import os
from datetime import datetime
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from config import SERVER, CAMERA, FACE_DETECTION, UI, PATHS, ensure_directories
import traceback
import logging
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.spinner import Spinner
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
from kivy.utils import get_color_from_hex

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("client_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FaceAuthClient")

class FaceAuthClient(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("Initializing FaceAuthClient")
        
        # Create tabbed panel
        self.tabs = TabbedPanel(do_default_tab=False, tab_width=Window.width / 3)
        self.tabs.background_color = get_color_from_hex("#f0f0f0")  # Light background
        
        # Authentication Tab
        auth_tab = TabbedPanelItem(text='Authentication')
        auth_tab.text_size = (self.tabs.tab_width, None)
        auth_tab.halign = 'center'

        auth_layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        # Camera preview
        self.image = Image()
        auth_layout.add_widget(self.image)
        
        # Status label
        # self.status_label = Label(
        #     text='Initializing...',
        #     size_hint_y=None,
        #     height=UI['status_bar_height']
        # )

        self.status_label = Label(
            text='Initializing...',
            font_size='16sp',
            color=get_color_from_hex("#333333"),
            size_hint_y=None,
            height=UI['status_bar_height']
        )
        auth_layout.add_widget(self.status_label)
        
        # Buttons
        # self.capture_doctor_btn = Button(
        #     text='Capture Doctor Image',
        #     size_hint_y=None,
        #     height=UI['button_height']
        # )

        self.capture_doctor_btn = Button(
            text='Capture Doctor Image',
            font_size='16sp',
            background_color=get_color_from_hex("#00a63e"),
            color=get_color_from_hex("#ffffff"),
            size_hint_y=None,
            height=UI['button_height']
        )
        
        self.capture_doctor_btn.bind(on_press=self.capture_doctor)
        auth_layout.add_widget(self.capture_doctor_btn)
        
        self.capture_patient_btn = Button(
            text='Capture Patient Image',
            size_hint_y=None,
            height=UI['button_height']
        )
        self.capture_patient_btn.bind(on_press=self.capture_patient)
        auth_layout.add_widget(self.capture_patient_btn)
        
        self.authenticate_btn = Button(
            text='Authenticate',
            size_hint_y=None,
            height=UI['button_height']
        )
        self.authenticate_btn.bind(on_press=self.authenticate)
        auth_layout.add_widget(self.authenticate_btn)
        
        auth_tab.add_widget(auth_layout)
        self.tabs.add_widget(auth_tab)
        
        # Registration Tab
        reg_tab = TabbedPanelItem(text='Register New User')
        reg_tab.text_size = (self.tabs.tab_width, None)
        reg_tab.halign = 'center'

        reg_layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        # Camera preview for registration
        self.reg_image = Image()
        reg_layout.add_widget(self.reg_image)
        
        # Status label for registration
        self.reg_status_label = Label(
            text='Ready to register',
            size_hint_y=None,
            height=UI['status_bar_height']
        )
        reg_layout.add_widget(self.reg_status_label)
        
        # Name input
        # self.name_input = TextInput(
        #     hint_text='Enter your name',
        #     size_hint_y=None,
        #     height=UI['button_height']
        # )

        self.name_input = TextInput(
            hint_text='Enter your name',
            font_size='16sp',
            padding=[10, 10, 10, 10],
            size_hint_y=None,
            height=UI['button_height']
        )
        
        reg_layout.add_widget(self.name_input)
        
        # Role selection
        self.role_selector = BoxLayout(orientation='horizontal', size_hint_y=None, height=UI['button_height'])
        self.role_selector.add_widget(Label(text='Select Role:'))
        
        self.role_buttons = BoxLayout(orientation='horizontal', size_hint_y=None, height=UI['button_height'])
        self.doctor_radio = ToggleButton(text='Doctor', group='role', state='down')
        self.patient_radio = ToggleButton(text='Patient', group='role')
        self.role_buttons.add_widget(self.doctor_radio)
        self.role_buttons.add_widget(self.patient_radio)
        self.role_selector.add_widget(self.role_buttons)
        reg_layout.add_widget(self.role_selector)
        
        # Capture and Register buttons
        self.capture_reg_btn = Button(
            text='Capture Image',
            size_hint_y=None,
            height=UI['button_height']
        )
        self.capture_reg_btn.bind(on_press=self.capture_registration)
        reg_layout.add_widget(self.capture_reg_btn)
        
        self.register_btn = Button(
            text='Register',
            size_hint_y=None,
            height=UI['button_height']
        )
        self.register_btn.bind(on_press=self.register_user)
        reg_layout.add_widget(self.register_btn)
        
        reg_tab.add_widget(reg_layout)
        self.tabs.add_widget(reg_tab)
        
        # Access Management Tab
        access_tab = TabbedPanelItem(text='Access Management')
        access_tab.text_size = (self.tabs.tab_width, None)
        access_tab.halign = 'center'
        access_layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        # Doctor selection
        doctor_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=UI['button_height'])
        doctor_layout.add_widget(Label(text='Select Doctor:'))
        self.doctor_spinner = Spinner(
            text='Select Doctor',
            values=[],
            size_hint_y=None,
            height=UI['button_height']
        )
        doctor_layout.add_widget(self.doctor_spinner)
        access_layout.add_widget(doctor_layout)
        
        # Patient selection
        patient_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=UI['button_height'])
        patient_layout.add_widget(Label(text='Select Patient:'))
        self.patient_spinner = Spinner(
            text='Select Patient',
            values=[],
            size_hint_y=None,
            height=UI['button_height']
        )
        patient_layout.add_widget(self.patient_spinner)
        access_layout.add_widget(patient_layout)
        
        # Date selection
        date_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=UI['button_height'])
        date_layout.add_widget(Label(text='Select Date:'))
        self.date_input = TextInput(
            hint_text='YYYY-MM-DD',
            size_hint_y=None,
            height=UI['button_height'],
            multiline=False
        )
        date_layout.add_widget(self.date_input)
        access_layout.add_widget(date_layout)
        
        # Visit type selection
        visit_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=UI['button_height'])
        visit_layout.add_widget(Label(text='Visit Type:'))
        self.visit_spinner = Spinner(
            text='regular',
            values=['regular', 'emergency', 'follow-up'],
            size_hint_y=None,
            height=UI['button_height']
        )
        visit_layout.add_widget(self.visit_spinner)
        access_layout.add_widget(visit_layout)
        
        # Status label
        self.access_status_label = Label(
            text='Ready to create access',
            size_hint_y=None,
            height=UI['status_bar_height']
        )
        access_layout.add_widget(self.access_status_label)
        
        # Create access button
        self.create_access_btn = Button(
            text='Create Access',
            size_hint_y=None,
            height=UI['button_height']
        )
        self.create_access_btn.bind(on_press=lambda x: asyncio.run(self._create_access()))
        access_layout.add_widget(self.create_access_btn)
        
        # Refresh button
        self.refresh_btn = Button(
            text='Refresh Lists',
            size_hint_y=None,
            height=UI['button_height']
        )
        self.refresh_btn.bind(on_press=self.refresh_lists)
        access_layout.add_widget(self.refresh_btn)
        
        access_tab.add_widget(access_layout)
        self.tabs.add_widget(access_tab)
        
        self.add_widget(self.tabs)
        
        # Initialize state
        self.doctor_image = None
        self.patient_image = None
        self.registration_image = None
        self.server_url = SERVER['url']
        
        # Ensure directories exist
        ensure_directories()
        
        # Initialize camera
        logger.debug("Initializing camera")
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            error_msg = "Error: Could not open camera"
            logger.error(error_msg)
            self.status_label.text = error_msg
            return
            
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        self.capture.set(cv2.CAP_PROP_FPS, CAMERA['fps'])
        logger.info(f"Camera initialized with resolution {CAMERA['width']}x{CAMERA['height']} at {CAMERA['fps']} FPS")
        
        # Initialize YOLO model
        logger.debug("Loading YOLO model")
        try:
            model_path = os.path.join(os.path.dirname(__file__), FACE_DETECTION['model_path'])
            logger.debug(f"Model path: {model_path}")
            
            if not os.path.exists(model_path):
                error_msg = f"Error: YOLO model file not found at {model_path}"
                logger.error(error_msg)
                self.status_label.text = error_msg
                return
                
            self.model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")
            self.status_label.text = 'Ready'
            
        except Exception as e:
            error_msg = f"Error: Failed to load face detection model: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.status_label.text = error_msg
            return
        
        # Start camera update
        Clock.schedule_interval(self.update_camera, 1.0/CAMERA['fps'])
        
        # Set window properties
        Window.size = UI['window_size']
        Window.minimum_width, Window.minimum_height = UI['window_size']
        logger.info("FaceAuthClient initialization complete")

    def update_camera(self, dt):
        """Update camera preview."""
        try:
            ret, frame = self.capture.read()
            if ret:
                # Convert frame to texture
                buf = cv2.flip(frame, 0).tobytes()
                
                # Update both previews
                texture = self.image.texture
                if texture is None or texture.width != frame.shape[1] or texture.height != frame.shape[0]:
                    texture = self.image.texture = Texture.create(
                        size=(frame.shape[1], frame.shape[0]), colorfmt='bgr'
                    )
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.image.canvas.ask_update()
                
                # Update registration preview
                reg_texture = self.reg_image.texture
                if reg_texture is None or reg_texture.width != frame.shape[1] or reg_texture.height != frame.shape[0]:
                    reg_texture = self.reg_image.texture = Texture.create(
                        size=(frame.shape[1], frame.shape[0]), colorfmt='bgr'
                    )
                reg_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.reg_image.canvas.ask_update()
        except Exception as e:
            logger.error(f"Error updating camera: {e}")
            traceback.print_exc()

    def capture_face(self, instance):
        """Capture face image and run YOLO detection."""
        try:
            logger.debug("Capturing face image")
            ret, frame = self.capture.read()
            if not ret:
                logger.error("Failed to capture frame")
                self.status_label.text = "Error: Failed to capture image"
                return
            
            # Run YOLO face detection
            logger.debug("Running YOLO face detection")
            results = self.model(frame)
            
            # Draw rectangles around detected faces
            frame_with_boxes = frame.copy()
            face_detected = False
            face_region = None
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw rectangle
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Extract face region with padding
                    padding = 20
                    h, w = frame.shape[:2]
                    x1_padded = max(0, x1 - padding)
                    y1_padded = max(0, y1 - padding)
                    x2_padded = min(w, x2 + padding)
                    y2_padded = min(h, y2 + padding)
                    face_region = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                    face_detected = True
            
            # Update camera preview with boxes
            buf = cv2.flip(frame_with_boxes, 0).tobytes()
            texture = Texture.create(size=(frame_with_boxes.shape[1], frame_with_boxes.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture
            
            if not face_detected:
                logger.warning("No face detected in frame")
                self.status_label.text = "No face detected. Please position your face in the frame."
                return
            
            if face_region is None or face_region.size == 0:
                logger.error("Invalid face region extracted")
                self.status_label.text = "Error: Failed to extract face region"
                return
            
            # Store the face region based on current step
            if instance == self.capture_doctor_btn:
                logger.info("Storing doctor face image")
                self.doctor_image = face_region
                self.status_label.text = 'Doctor image captured'
            else:
                logger.info("Storing patient face image")
                self.patient_image = face_region
                self.status_label.text = 'Patient image captured'
            
        except Exception as e:
            logger.error(f"Error capturing face: {e}")
            logger.error(traceback.format_exc())
            self.status_label.text = f"Error: {str(e)}"

    def capture_doctor(self, instance):
        """Capture doctor's face."""
        self.capture_face(self.capture_doctor_btn)

    def capture_patient(self, instance):
        """Capture patient's face."""
        self.capture_face(self.capture_patient_btn)

    async def send_authentication_request(self):
        """Send authentication request to server."""
        logger.info("Starting authentication process")
        
        if self.doctor_image is None or self.patient_image is None:
            logger.warning("Missing doctor or patient image")
            self.status_label.text = 'Please capture both doctor and patient images'
            return
        
        try:
            logger.debug(f"Connecting to server at {self.server_url}")
            async with websockets.connect(self.server_url) as websocket:
                logger.info("Connected to server")
                
                # Prepare images for sending
                logger.debug("Encoding images to base64")
                doctor_base64 = base64.b64encode(cv2.imencode('.jpg', self.doctor_image)[1]).decode()
                patient_base64 = base64.b64encode(cv2.imencode('.jpg', self.patient_image)[1]).decode()
                
                # Send both images in a single message
                message = {
                    'type': 'authentication',
                    'doctor_image': doctor_base64,
                    'patient_image': patient_base64
                }
                
                logger.debug("Sending authentication request")
                await websocket.send(json.dumps(message))
                
                logger.debug("Waiting for server response")
                response = json.loads(await websocket.recv())
                logger.info(f"Received response: {response}")
                
                if response['status'] == 'error':
                    logger.warning(f"Authentication failed: {response['message']}")
                    self.status_label.text = f'Authentication failed: {response["message"]}'
                else:
                    logger.info(f"Authentication successful: {response['message']}")
                    self.status_label.text = f'Authentication successful: {response["message"]}'
                
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            traceback.print_exc()
            self.status_label.text = f'Error: {str(e)}'

    def authenticate(self, instance):
        """Start authentication process."""
        logger.info("Authenticate button pressed")
        asyncio.run(self.send_authentication_request())

    def capture_registration(self, instance):
        """Capture face image for registration."""
        try:
            logger.debug("Capturing registration image")
            ret, frame = self.capture.read()
            if not ret:
                logger.error("Failed to capture frame")
                self.reg_status_label.text = "Error: Failed to capture image"
                return
            
            # Run YOLO face detection
            logger.debug("Running YOLO face detection")
            results = self.model(frame)
            
            # Draw rectangles around detected faces
            frame_with_boxes = frame.copy()
            face_detected = False
            face_region = None
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw rectangle
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Extract face region with padding
                    padding = 20
                    h, w = frame.shape[:2]
                    x1_padded = max(0, x1 - padding)
                    y1_padded = max(0, y1 - padding)
                    x2_padded = min(w, x2 + padding)
                    y2_padded = min(h, y2 + padding)
                    face_region = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                    face_detected = True
            
            # Update registration preview with boxes
            buf = cv2.flip(frame_with_boxes, 0).tobytes()
            texture = Texture.create(size=(frame_with_boxes.shape[1], frame_with_boxes.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.reg_image.texture = texture
            
            if not face_detected:
                logger.warning("No face detected in frame")
                self.reg_status_label.text = "No face detected. Please position your face in the frame."
                return
            
            if face_region is None or face_region.size == 0:
                logger.error("Invalid face region extracted")
                self.reg_status_label.text = "Error: Failed to extract face region"
                return
            
            self.registration_image = face_region
            self.reg_status_label.text = 'Face captured successfully'
            
        except Exception as e:
            logger.error(f"Error capturing registration image: {e}")
            logger.error(traceback.format_exc())
            self.reg_status_label.text = f"Error: {str(e)}"

    async def send_registration_request(self):
        """Send registration request to server."""
        logger.info("Starting registration process")
        
        if self.registration_image is None:
            logger.warning("No registration image captured")
            self.reg_status_label.text = 'Please capture an image first'
            return
            
        if not self.name_input.text.strip():
            logger.warning("No name provided")
            self.reg_status_label.text = 'Please enter your name'
            return
            
        # Determine role
        role = 'Doctor' if self.doctor_radio.state == 'down' else 'Patient'
        
        try:
            logger.debug(f"Connecting to server at {self.server_url}")
            async with websockets.connect(self.server_url) as websocket:
                logger.info("Connected to server")
                
                # Prepare image for sending
                logger.debug("Encoding image to base64")
                image_base64 = base64.b64encode(cv2.imencode('.jpg', self.registration_image)[1]).decode()
                
                # Send registration request
                message = {
                    'type': 'registration',
                    'image': image_base64,
                    'name': self.name_input.text.strip(),
                    'role': role
                }
                
                logger.debug("Sending registration request")
                await websocket.send(json.dumps(message))
                
                logger.debug("Waiting for server response")
                response = json.loads(await websocket.recv())
                logger.info(f"Received response: {response}")
                
                if response['status'] == 'error':
                    logger.warning(f"Registration failed: {response['message']}")
                    self.reg_status_label.text = f'Registration failed: {response["message"]}'
                else:
                    logger.info(f"Registration successful: {response['message']}")
                    self.reg_status_label.text = f'Registration successful: {response["message"]}'
                    # Clear form
                    self.name_input.text = ''
                    self.registration_image = None
                
        except Exception as e:
            logger.error(f"Error during registration: {e}")
            traceback.print_exc()
            self.reg_status_label.text = f'Error: {str(e)}'

    def register_user(self, instance):
        """Start registration process."""
        logger.info("Register button pressed")
        asyncio.run(self.send_registration_request())

    async def get_doctors_list(self):
        """Get list of doctors from server."""
        try:
            logger.debug(f"Connecting to server at {self.server_url}")
            async with websockets.connect(self.server_url) as websocket:
                logger.info("Connected to server")
                
                # Request doctors list
                message = {'type': 'get_doctors'}
                await websocket.send(json.dumps(message))
                
                # Get response
                response = json.loads(await websocket.recv())
                logger.info(f"Received response: {response}")
                
                if response['status'] == 'error':
                    logger.warning(f"Failed to get doctors list: {response['message']}")
                    return []
                
                # Format doctors for spinner
                doctors = [f"{doc['name']} ({doc['id']})" for doc in response['doctors']]
                return doctors
                
        except Exception as e:
            logger.error(f"Error getting doctors list: {e}")
            traceback.print_exc()
            return []

    async def get_patients_list(self):
        """Get list of patients from server."""
        try:
            logger.debug(f"Connecting to server at {self.server_url}")
            async with websockets.connect(self.server_url) as websocket:
                logger.info("Connected to server")
                
                # Request patients list
                message = {'type': 'get_patients'}
                await websocket.send(json.dumps(message))
                
                # Get response
                response = json.loads(await websocket.recv())
                logger.info(f"Received response: {response}")
                
                if response['status'] == 'error':
                    logger.warning(f"Failed to get patients list: {response['message']}")
                    return []
                
                # Format patients for spinner
                patients = [f"{pat['name']} ({pat['id']})" for pat in response['patients']]
                return patients
                
        except Exception as e:
            logger.error(f"Error getting patients list: {e}")
            traceback.print_exc()
            return []

    def refresh_lists(self, instance):
        """Refresh doctor and patient lists."""
        logger.info("Refreshing lists")
        self.access_status_label.text = 'Refreshing lists...'
        asyncio.run(self._refresh_lists())

    async def _refresh_lists(self):
        """Internal method to refresh lists."""
        try:
            # Get updated lists
            doctors = await self.get_doctors_list()
            patients = await self.get_patients_list()
            
            # Update spinners
            self.doctor_spinner.values = doctors
            self.patient_spinner.values = patients
            
            if doctors:
                self.doctor_spinner.text = doctors[0]
            if patients:
                self.patient_spinner.text = patients[0]
            
            self.access_status_label.text = 'Lists refreshed successfully'
            
        except Exception as e:
            logger.error(f"Error refreshing lists: {e}")
            traceback.print_exc()
            self.access_status_label.text = f'Error refreshing lists: {str(e)}'

    async def _create_access(self):
        """Create new access entry."""
        try:
            logger.info("Creating access entry")
            
            # Validate inputs
            if self.doctor_spinner.text == 'Select Doctor':
                self.access_status_label.text = 'Please select a doctor'
                return
            if self.patient_spinner.text == 'Select Patient':
                self.access_status_label.text = 'Please select a patient'
                return
            if not self.date_input.text.strip():
                self.access_status_label.text = 'Please enter a date'
                return
            
            # Extract IDs from spinner text
            doctor_id = self.doctor_spinner.text.split('(')[1].strip(')')
            patient_id = self.patient_spinner.text.split('(')[1].strip(')')
            
            # Prepare request
            message = {
                'type': 'create_access',
                'doctor_id': doctor_id,
                'patient_id': patient_id,
                'access_date': self.date_input.text.strip(),
                'visit_type': self.visit_spinner.text
            }
            
            # Send request
            logger.debug(f"Connecting to server at {self.server_url}")
            async with websockets.connect(self.server_url) as websocket:
                logger.info("Connected to server")
                await websocket.send(json.dumps(message))
                
                # Get response
                response = json.loads(await websocket.recv())
                logger.info(f"Received response: {response}")
                
                if response['status'] == 'error':
                    logger.warning(f"Failed to create access: {response['message']}")
                    self.access_status_label.text = f'Error: {response["message"]}'
                else:
                    logger.info("Access created successfully")
                    self.access_status_label.text = 'Access created successfully'
                    # Clear date input
                    self.date_input.text = ''
                
        except Exception as e:
            logger.error(f"Error creating access: {e}")
            traceback.print_exc()
            self.access_status_label.text = f'Error: {str(e)}'

class FaceAuthApp(App):
    def build(self):
        logger.info("Building FaceAuthApp")
        return FaceAuthClient()
    
    def on_stop(self):
        """Clean up when app is closed."""
        logger.info("Application stopping, cleaning up resources")
        if hasattr(self.root, 'capture') and self.root.capture is not None:
            self.root.capture.release()
            logger.info("Camera released")
        return super().on_stop()

if __name__ == '__main__':
    logger.info("Starting Face Authentication Client")
    FaceAuthApp().run()
