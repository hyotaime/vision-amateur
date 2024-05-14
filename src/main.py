import cv2 as cv
import pyautogui

import common
import pinch
import hand_pipe

hp = hand_pipe.Handpipe()

# Mediapipe 구성요소 초기화
screen_width, screen_height = pyautogui.size()

# 기본 웹캠 캡처 객체 초기화
cap = cv.VideoCapture(0)
# Click
mouse_clicked = False
# Zoom
initial_distance = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 웹캠 영상을 거울 모드로 뒤집기
    frame = cv.flip(frame, 1)
    results = hp.process_frame(frame)
    # 다시 BGR로 변환하여 OpenCV로 표시 가능하게 함
    image = frame

    num_touching_hands = 0

    if results.multi_hand_landmarks:
        # 두 손이 검출되면
        if len(results.multi_hand_landmarks) == 2:
            # 양손의 랜드마크 가져오기
            hand1, hand2 = results.multi_hand_landmarks
            # 손 랜드마크 그리기
            hp.draw_landmarks(image, hand1)
            hp.draw_landmarks(image, hand2)

            if pinch.is_pinch(hand1, image) and pinch.is_pinch(hand2, image):
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
            hp.draw_landmarks(image, hand_landmarks)
            hand_pos = common.get_hand_avg_pos(hand_landmarks, image.shape)
            # 마우스 커서를 화면 전체 크기로 비율 조정 후 이동
            screen_x = screen_width * hand_pos[0] / image.shape[1]
            screen_y = screen_height * hand_pos[1] / image.shape[0]
            pyautogui.moveTo(screen_x, screen_y)

            # 손가락이 닿은 상태라면 마우스 클릭
            if pinch.is_pinch(hand_landmarks, image) and not mouse_clicked:
                pyautogui.mouseDown()
                print('Mouse Down!')
                mouse_clicked = True
            # 손가락이 떨어졌다면 마우스 클릭 해제
            elif not pinch.is_pinch(hand_landmarks, image) and mouse_clicked:
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
