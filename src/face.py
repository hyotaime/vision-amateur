import cv2 as cv
import numpy as np
import mediapipe as mp


def eye_center(frame, face_landmarks, LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS):
    """
    Calculate the center of the left and right eyes, then find the average eye center.
    Draw circles at each center point on the frame.

    Args:
    frame (numpy array): The frame containing the face.
    face_landmarks (object): The detected face landmarks.
    LEFT_EYE_LANDMARKS (list): Indices of landmarks corresponding to the left eye.
    RIGHT_EYE_LANDMARKS (list): Indices of landmarks corresponding to the right eye.

    Returns:
    tuple: The x, y coordinates of the average eye center.
    """
    # Get frame dimensions
    height, width, _ = frame.shape

    # Convert landmarks to pixel coordinates
    landmarks = [(int(p.x * width), int(p.y * height)) for p in face_landmarks.landmark]

    # Calculate the center of the left eye
    left_eye_center = np.mean([landmarks[i] for i in LEFT_EYE_LANDMARKS], axis=0).astype(int)
    cv.circle(frame, tuple(left_eye_center), 3, (0, 255, 0), -1)

    # Calculate the center of the right eye
    right_eye_center = np.mean([landmarks[i] for i in RIGHT_EYE_LANDMARKS], axis=0).astype(int)
    cv.circle(frame, tuple(right_eye_center), 3, (0, 255, 0), -1)

    # Calculate the average center of both eyes
    center = np.mean([left_eye_center, right_eye_center], axis=0).astype(int)
    cv.circle(frame, tuple(center), 3, (255, 0, 0), -1)

    return int(center[0]), int(center[1])


def calculate_ear(eye):
    """
    Calculate the Eye Aspect Ratio (EAR) for an eye.

    Args:
    eye (numpy array): The coordinates of the eye landmarks.

    Returns:
    float: The calculated EAR value.
    """
    # EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    A = np.linalg.norm(eye[1] - eye[5])  # Distance between vertical landmarks
    B = np.linalg.norm(eye[2] - eye[4])  # Distance between vertical landmarks
    C = np.linalg.norm(eye[0] - eye[3])  # Distance between horizontal landmarks
    ear = (A + B) / (2.0 * C)
    return ear


def get_eye_landmarks(landmarks, eye_landmarks):
    """
    Get the coordinates of the eye landmarks from the overall face landmarks.

    Args:
    landmarks (list): List of all face landmarks.
    eye_landmarks (list): Indices of the landmarks corresponding to an eye.

    Returns:
    numpy array: Coordinates of the eye landmarks.
    """
    return np.array([landmarks[i] for i in eye_landmarks])


class Face:
    def __init__(self):
        """
        Initialize the Face class with Mediapipe's face mesh solution.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()

    def process_rgb(self, frame):
        """
        Process an input frame to detect face landmarks.

        Args:
        frame (numpy array): The frame to process.

        Returns:
        object: The detected landmarks by Mediapipe.
        """
        # Convert BGR to RGB for Mediapipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Process the frame to detect landmarks
        results = self.face_mesh.process(rgb_frame)
        return results
