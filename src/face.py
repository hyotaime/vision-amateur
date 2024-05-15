import cv2 as cv
import numpy as np
import mediapipe as mp


def eye_center(frame, face_landmarks, LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS):
    height, width, _ = frame.shape
    landmarks = [(int(p.x * width), int(p.y * height)) for p in face_landmarks.landmark]

    # 왼쪽 눈 중심 계산
    left_eye_center = np.mean([landmarks[i] for i in LEFT_EYE_LANDMARKS], axis=0).astype(int)
    cv.circle(frame, tuple(left_eye_center), 3, (0, 255, 0), -1)

    # 오른쪽 눈 중심 계산
    right_eye_center = np.mean([landmarks[i] for i in RIGHT_EYE_LANDMARKS], axis=0).astype(int)
    cv.circle(frame, tuple(right_eye_center), 3, (0, 255, 0), -1)

    # 평균 눈 중심 계산
    eye_center = np.mean([left_eye_center, right_eye_center], axis=0).astype(int)
    cv.circle(frame, tuple(eye_center), 3, (255, 0, 0), -1)

    return int(eye_center[0]), int(eye_center[1])


class Face:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()

    def process_rgb(self, frame):
        """
        Process an input frame to detect face landmarks.
        Returns the landmarks detected by Mediapipe.
        """
        # Convert BGR to RGB for Mediapipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        return results
