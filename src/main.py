import cv2 as cv
import pyautogui

import common
import hand
import face

# Constants
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144, 145]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380, 374]
EYE_AR_THRESH = 0.2

mouse_clicked = False
initial_distance = None


def draw_center_rectangle(frame):
    """
    Draw a rectangle in the center of the frame.
    """
    height, width, _ = frame.shape
    rect_width, rect_height = width // 4, height // 4
    center_x, center_y = width // 2, height // 2
    top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
    bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
    cv.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)


def process_face_landmarks(face_results, frame):
    """
    Process face landmarks and handle blink detection and mouse control.
    """
    height, width, _ = frame.shape

    for face_landmarks in face_results.multi_face_landmarks:
        landmarks = [(int(p.x * width), int(p.y * height)) for p in face_landmarks.landmark]

        left_eye = face.get_eye_landmarks(landmarks, LEFT_EYE_LANDMARKS)
        right_eye = face.get_eye_landmarks(landmarks, RIGHT_EYE_LANDMARKS)

        left_ear = face.calculate_ear(left_eye)
        right_ear = face.calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EYE_AR_THRESH:
            print('Blink')
            pyautogui.click()

        cv.putText(frame, f"EAR: {ear:.2f}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        eye_center_pos = face.eye_center(frame, face_landmarks, LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS)
        screen_x = (screen_width / 2) + (eye_center_pos[0] - (frame.shape[1] / 2)) * 4
        screen_y = (screen_height / 2) + (eye_center_pos[1] - (frame.shape[0] / 2)) * 4
        pyautogui.moveTo(screen_x, screen_y)


def process_hand_landmarks(results, frame):
    """
    Process hand landmarks and handle pinch gestures and zoom control.
    """
    global mouse_clicked, initial_distance

    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 2:
            handle_two_hands(results.multi_hand_landmarks, frame)
        elif len(results.multi_hand_landmarks) == 1:
            handle_single_hand(results.multi_hand_landmarks[0], frame)
        else:
            initial_distance = None


def handle_two_hands(hands, frame):
    """
    Handle gestures and actions when two hands are detected.
    """
    global initial_distance

    hand1, hand2 = hands
    myhand.draw_landmarks(frame, hand1)
    myhand.draw_landmarks(frame, hand2)

    if hand.is_pinch(hand1, frame) and hand.is_pinch(hand2, frame):
        cv.putText(frame, "Zoom", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        current_distance = common.calculate_distance(hand1, hand2)

        if initial_distance is None:
            initial_distance = current_distance
        else:
            zoom_scale = current_distance / initial_distance
            pyautogui.hotkey('command', '+') if zoom_scale > 1 else pyautogui.hotkey('command', '-')


def handle_single_hand(hand_landmarks, frame):
    """
    Handle gestures and actions when a single hand is detected.
    """
    global mouse_clicked

    myhand.draw_landmarks(frame, hand_landmarks)

    if hand.is_pinch(hand_landmarks, frame) and not mouse_clicked:
        pyautogui.mouseDown()
        print('Mouse Down!')
        mouse_clicked = True
    elif not hand.is_pinch(hand_landmarks, frame) and mouse_clicked:
        pyautogui.mouseUp()
        print('Mouse Up!')
        mouse_clicked = False


if __name__ == '__main__':
    myface = face.Face()
    myhand = hand.Hand()
    screen_width, screen_height = pyautogui.size()
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        draw_center_rectangle(frame)

        hand_results = myhand.process_rgb(frame)
        face_results = myface.process_rgb(frame)

        if face_results.multi_face_landmarks:
            process_face_landmarks(face_results, frame)

        if hand_results.multi_hand_landmarks:
            process_hand_landmarks(hand_results, frame)

        cv.imshow('Vision Amateur', frame)

        if cv.waitKey(1) == 27:
            break

    cap.release()
    cv.destroyAllWindows()
