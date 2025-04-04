# Face Authentication System

A secure face recognition-based authentication system for healthcare environments, allowing doctors to authenticate patients and manage access permissions.

## Overview

This application provides a secure way to authenticate patients using facial recognition technology. It consists of two main components:

1. **Client Application**: A desktop application built with Kivy that provides a user interface for capturing faces, registering users, and authenticating patients.
2. **Server Application**: A WebSocket server that handles face recognition, user registration, and access management.

## Features

- **Face Detection**: Uses YOLOv8 for accurate face detection
- **Face Recognition**: Implements FaceNet for facial recognition
- **User Registration**: Register doctors and patients with facial data
- **Authentication**: Verify patient identity using facial recognition
- **Access Management**: Create and manage access permissions for doctor-patient interactions
- **Real-time Camera Preview**: Live camera feed with face detection visualization

## System Requirements

### Client Requirements
- Python 3.8+
- Webcam
- OpenCV
- Kivy
- YOLOv8
- WebSockets

### Server Requirements
- Python 3.8+
- MySQL Database
- OpenCV
- DeepFace
- WebSockets

## Installation

### Client Setup

1. Navigate to the client directory:
   ```
   cd new_client
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```
     source venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Ensure the YOLOv8 face detection model is in the correct location:
   ```
   yolov8n-face-lindevs.pt
   ```

### Server Setup

1. Navigate to the server directory:
   ```
   cd new_server
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```
     source venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Set up the MySQL database:
   - Create a database named `face_auth_db`
   - Update the database credentials in `config.py` if needed

## Configuration

### Client Configuration

Edit `new_client/config.py` to configure:
- Server connection details
- Camera settings
- Face detection parameters
- UI settings

### Server Configuration

Edit `new_server/config.py` to configure:
- Database connection details
- Face recognition parameters
- Server settings
- Debug options

## Usage

### Starting the Server

1. Navigate to the server directory:
   ```
   cd new_server
   ```

2. Activate the virtual environment:
   ```
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. Start the server:
   ```
   python server.py
   ```

### Starting the Client

1. Navigate to the client directory:
   ```
   cd new_client
   ```

2. Activate the virtual environment:
   ```
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. Start the client:
   ```
   python client.py
   ```

## Application Workflow

1. **Registration**:
   - Select the "Register New User" tab
   - Enter the user's name
   - Select the role (Doctor or Patient)
   - Capture the user's face image
   - Click "Register" to save the user

2. **Authentication**:
   - Select the "Authentication" tab
   - Capture the doctor's face image
   - Capture the patient's face image
   - Click "Authenticate" to verify the patient's identity

3. **Access Management**:
   - Select the "Access Management" tab
   - Select a doctor from the dropdown
   - Select a patient from the dropdown
   - Enter the access date
   - Select the visit type
   - Click "Create Access" to create a new access entry

## Troubleshooting

- **Camera not working**: Ensure your webcam is properly connected and not being used by another application
- **Face detection issues**: Make sure the YOLOv8 model file is in the correct location
- **Connection errors**: Verify that the server is running and the connection details in the client config are correct
- **Database errors**: Check that the MySQL server is running and the credentials in the server config are correct

## Logging

Both the client and server applications generate log files:
- Client logs: `new_client/client_debug.log`
- Server logs: `new_server/server_debug.log`

These logs can be helpful for troubleshooting issues.
