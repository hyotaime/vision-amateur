import cv2 as cv
import mediapipe as mp
import pyautogui

# Mediapipe 구성요소 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

# 기본 웹캠 캡처 객체 초기화
cap = cv.VideoCapture(0)


def is_thumb_index_touching(hand_landmarks):
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
    return distance < 50


def get_hand_avg_pos(hand_landmarks, frame_shape):
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    thumb_tip = hand_landmarks.landmark[THUMB_TIP]
    index_tip = hand_landmarks.landmark[INDEX_TIP]
    middle_tip = hand_landmarks.landmark[MIDDLE_TIP]
    ring_tip = hand_landmarks.landmark[RING_TIP]
    pinky_tip = hand_landmarks.landmark[PINKY_TIP]

    # 평균 좌표 계산
    avg_x = (thumb_tip.x + index_tip.x + middle_tip.x + ring_tip.x + pinky_tip.x) / 5
    avg_y = (thumb_tip.y + index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y) / 5

    # 이미지 좌표로 변환
    return int(avg_x * frame_shape[1]), int(avg_y * frame_shape[0])


mouse_clicked = False

# Zoom
initial_distance = None


def calculate_distance(hand1, hand2):
    """
    두 손의 검지 랜드마크를 사용하여 거리 계산
    """
    INDEX_TIP = 8
    hand1_index_tip = hand1.landmark[INDEX_TIP]
    hand2_index_tip = hand2.landmark[INDEX_TIP]

    # 거리 계산
    dx = hand1_index_tip.x - hand2_index_tip.x
    dy = hand1_index_tip.y - hand2_index_tip.y

    return (dx ** 2 + dy ** 2) ** 0.5


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 웹캠 영상을 거울 모드로 뒤집기
    frame = cv.flip(frame, 1)

    # BGR 이미지를 RGB로 변환 (Mediapipe는 RGB 이미지를 필요로 함)
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Mediapipe Hands를 사용하여 손 랜드마크 검출
    results = hands.process(image)

    # 다시 BGR로 변환하여 OpenCV로 표시 가능하게 함
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    num_touching_hands = 0

    if results.multi_hand_landmarks:
        # 두 손이 검출되면
        if len(results.multi_hand_landmarks) == 2:
            # 양손의 랜드마크 가져오기
            hand1, hand2 = results.multi_hand_landmarks
            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(image, hand1, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, hand2, mp_hands.HAND_CONNECTIONS)

            if is_thumb_index_touching(hand1) and is_thumb_index_touching(hand2):
                cv.putText(image, "Zoom", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # 두 손의 거리를 계산
                current_distance = calculate_distance(hand1, hand2)

                if initial_distance is None:
                    # 초기 거리 설정
                    initial_distance = current_distance

                else:
                    # 현재 거리와 초기 거리의 비율로 확대/축소 비율 계산
                    zoom_scale = current_distance / initial_distance
                    # 확대/축소 동작 수행
                    pyautogui.hotkey('command', '+') if zoom_scale > 1 else pyautogui.hotkey('command', '-')
        else:
            initial_distance = None

        # 손이 하나만 검출되면
        if len(results.multi_hand_landmarks) == 1:
            # 손 랜드마크 가져오기
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_pos = get_hand_avg_pos(hand_landmarks, image.shape)

            # 마우스 커서를 화면 전체 크기로 비율 조정 후 이동
            screen_x = screen_width * hand_pos[0] / image.shape[1]
            screen_y = screen_height * hand_pos[1] / image.shape[0]
            pyautogui.moveTo(screen_x, screen_y)

            # 손가락이 닿았는지 확인
            if is_thumb_index_touching(hand_landmarks):
                cv.putText(image, "Click", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 마우스 클릭을 한 번만 수행하기 위한 플래그
                if not mouse_clicked:
                    pyautogui.click()
                    print('Click!')
                    mouse_clicked = True
            else:
                mouse_clicked = False

    # 결과 이미지 표시
    cv.imshow('Hand Detection', image)

    # ESC 키를 누르면 종료
    if cv.waitKey(1) == 27:
        break

# 자원 정리
cap.release()
cv.destroyAllWindows()
