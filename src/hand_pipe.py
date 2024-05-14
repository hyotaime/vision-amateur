import cv2 as cv
import mediapipe as mp


class Handpipe:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        """
        Process an input frame to detect hand landmarks.
        Returns the landmarks detected by Mediapipe.
        """
        # Convert BGR to RGB for Mediapipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results

    def process_BGR(self, frame):
        """
        Process an input frame to detect hand landmarks.
        Returns the landmarks detected by Mediapipe.
        """
        # Convert BGR to RGB for Mediapipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return rgb_frame

    def draw_landmarks(self, frame, landmarks):
        """
        Draw hand landmarks on the frame.
        """
        self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
