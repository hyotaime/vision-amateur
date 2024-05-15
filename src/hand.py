import cv2 as cv
import mediapipe as mp


def is_pinch(hand_landmarks, image):
    """
    Check if the thumb and index finger are pinching (touching).

    Args:
    hand_landmarks (object): The detected hand landmarks.
    image (numpy array): The image containing the hand.

    Returns:
    bool: True if the thumb and index finger are pinching, False otherwise.
    """
    LIMIT = 30  # Distance threshold to consider fingers touching
    THUMB_TIP = 4  # Index for thumb tip landmark
    INDEX_TIP = 8  # Index for index finger tip landmark

    # Get landmark coordinates for thumb and index finger tips
    thumb_tip = hand_landmarks.landmark[THUMB_TIP]
    index_tip = hand_landmarks.landmark[INDEX_TIP]

    # Convert normalized coordinates to image coordinates
    thumb_pos = (int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0]))
    index_pos = (int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0]))

    # Calculate the distance between the thumb and index finger tips
    distance = ((thumb_pos[0] - index_pos[0]) ** 2 + (thumb_pos[1] - index_pos[1]) ** 2) ** 0.5

    # Consider fingers touching if the distance is below the threshold
    return distance < LIMIT


class Hand:
    def __init__(self):
        """
        Initialize the Hand class with Mediapipe's hand detection and drawing utilities.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def process_rgb(self, frame):
        """
        Process an input frame to detect hand landmarks.

        Args:
        frame (numpy array): The frame to process.

        Returns:
        object: The detected hand landmarks by Mediapipe.
        """
        # Convert BGR to RGB for Mediapipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Process the frame to detect hand landmarks
        results = self.hands.process(rgb_frame)
        return results

    def draw_landmarks(self, frame, landmarks):
        """
        Draw hand landmarks on the frame.

        Args:
        frame (numpy array): The frame on which to draw.
        landmarks (object): The detected hand landmarks.
        """
        self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
