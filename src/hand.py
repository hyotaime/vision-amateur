import cv2 as cv
import mediapipe as mp


def is_pinch(hand_landmarks, image):
    LIMIT = 30
    # 엄지와 검지의 랜드마크를 이용하여 두 손가락이 닿았는지 확인
    THUMB_TIP = 4
    INDEX_TIP = 8

    # 랜드마크 좌표 가져오기
    thumb_tip = hand_landmarks.landmark[THUMB_TIP]
    index_tip = hand_landmarks.landmark[INDEX_TIP]

    # 좌표를 이미지 좌표로 변환
    thumb_pos = (int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0]))
    index_pos = (int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0]))

    # 두 좌표 간의 거리 계산
    distance = ((thumb_pos[0] - index_pos[0]) ** 2 + (thumb_pos[1] - index_pos[1]) ** 2) ** 0.5

    # 일정 거리 내에 있으면 접촉으로 간주
    return distance < LIMIT


class Hand:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def process_rgb(self, frame):
        """
        Process an input frame to detect hand landmarks.
        Returns the landmarks detected by Mediapipe.
        """
        # Convert BGR to RGB for Mediapipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results

    def draw_landmarks(self, frame, landmarks):
        """
        Draw hand landmarks on the frame.
        """
        self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
