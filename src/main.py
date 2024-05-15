import cv2 as cv
import pyautogui
import mediapipe as mp

import common
import hand
import face

myface = face.Face()
myhand = hand.Hand()

# Mediapipe 구성요소 초기화
screen_width, screen_height = pyautogui.size()

# 기본 웹캠 캡처 객체 초기화
cap = cv.VideoCapture(0)
# 얼굴 인식
mp_face_mesh = mp.solutions.face_mesh
# 눈 랜드마크 번호
LEFT_EYE_LANDMARKS = [33, 133, 160, 159, 158, 157, 173, 246]
RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 384, 398, 466]
# 손동작
mouse_clicked = False  # click
initial_distance = None  # zoom

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 웹캠 영상을 거울 모드로 뒤집기
    frame = cv.flip(frame, 1)
    # RGB로 변환
    results = myhand.process_rgb(frame)
    image = frame

    # 얼굴 이미지
    face_results = myface.process_rgb(frame)

    # 화면의 가운데를 중심으로 하고, 화면 길이의 절반을 변으로 가지는 직사각형 계산
    height, width, _ = frame.shape
    rect_width, rect_height = width // 4, height // 4
    center_x, center_y = width // 2, height // 2
    top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
    bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
    # 직사각형 그리기
    cv.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

    # 얼굴 랜드마크 검출
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            eye_center_pos = face.eye_center(image, face_landmarks, LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS)
            screen_x = (screen_width / 2) + (eye_center_pos[0] - (image.shape[1] / 2)) * 4
            screen_y = (screen_height / 2) + (eye_center_pos[1] - (image.shape[0] / 2)) * 4
            pyautogui.moveTo(screen_x, screen_y)

    num_touching_hands = 0

    if results.multi_hand_landmarks:
        # 두 손이 검출되면
        if len(results.multi_hand_landmarks) == 2:
            # 양손의 랜드마크 가져오기
            hand1, hand2 = results.multi_hand_landmarks
            # 손 랜드마크 그리기
            myhand.draw_landmarks(image, hand1)
            myhand.draw_landmarks(image, hand2)

            if hand.is_pinch(hand1, image) and hand.is_pinch(hand2, image):
                cv.putText(image, "Zoom", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # 두 손의 거리를 계산
                current_distance = common.calculate_distance(hand1, hand2)

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
            myhand.draw_landmarks(image, hand_landmarks)
            # 손가락이 닿은 상태라면 마우스 클릭
            if hand.is_pinch(hand_landmarks, image) and not mouse_clicked:
                pyautogui.mouseDown()
                print('Mouse Down!')
                mouse_clicked = True
            # 손가락이 떨어졌다면 마우스 클릭 해제
            elif not hand.is_pinch(hand_landmarks, image) and mouse_clicked:
                pyautogui.mouseUp()
                print('Mouse Up!')
                mouse_clicked = False

    # 결과 이미지 표시
    cv.imshow('Hand Detection', image)

    # ESC 키를 누르면 종료
    if cv.waitKey(1) == 27:
        break

# 자원 정리
cap.release()
cv.destroyAllWindows()
