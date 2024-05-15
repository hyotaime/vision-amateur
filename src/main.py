import cv2 as cv
import pyautogui
import mediapipe as mp

import common
import hand
import face

# Eye landmark indices
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144, 145]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380, 374]
# Blink detection threshold
EYE_AR_THRESH = 0.2
# Hand gesture flags
mouse_clicked = False  # Click status
initial_distance = None  # Zoom status

if __name__ == '__main__':
    myface = face.Face()
    myhand = hand.Hand()

    # Initialize Mediapipe components
    screen_width, screen_height = pyautogui.size()
    # Face mesh
    mp_face_mesh = mp.solutions.face_mesh
    # Initialize webcam capture object
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for mirror mode
        frame = cv.flip(frame, 1)
        # Process the frame for hand landmarks
        results = myhand.process_rgb(frame)
        image = frame

        # Process the frame for face landmarks
        face_results = myface.process_rgb(frame)

        # Calculate a rectangle in the center of the screen
        height, width, _ = frame.shape
        rect_width, rect_height = width // 4, height // 4
        center_x, center_y = width // 2, height // 2
        top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
        bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
        # Draw the rectangle
        cv.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

        # Detect face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                landmarks = [(int(p.x * width), int(p.y * height)) for p in face_landmarks.landmark]
                # Calculate left eye EAR
                left_eye = face.get_eye_landmarks(landmarks, LEFT_EYE_LANDMARKS)
                left_ear = face.calculate_ear(left_eye)

                # Calculate right eye EAR
                right_eye = face.get_eye_landmarks(landmarks, RIGHT_EYE_LANDMARKS)
                right_ear = face.calculate_ear(right_eye)

                # Calculate average EAR
                ear = (left_ear + right_ear) / 2.0

                if ear < EYE_AR_THRESH:
                    print('Blink')
                    pyautogui.click()  # Trigger mouse click event

                # Display EAR value on the frame
                cv.putText(frame, f"EAR: {ear:.2f}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Calculate eye center position and move mouse
                eye_center_pos = face.eye_center(image, face_landmarks, LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS)
                screen_x = (screen_width / 2) + (eye_center_pos[0] - (image.shape[1] / 2)) * 4
                screen_y = (screen_height / 2) + (eye_center_pos[1] - (image.shape[0] / 2)) * 4
                pyautogui.moveTo(screen_x, screen_y)

        num_touching_hands = 0

        if results.multi_hand_landmarks:
            # If two hands are detected
            if len(results.multi_hand_landmarks) == 2:
                # Get landmarks for both hands
                hand1, hand2 = results.multi_hand_landmarks
                # Draw hand landmarks
                myhand.draw_landmarks(image, hand1)
                myhand.draw_landmarks(image, hand2)

                if hand.is_pinch(hand1, image) and hand.is_pinch(hand2, image):
                    cv.putText(image, "Zoom", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Calculate the distance between the two hands
                    current_distance = common.calculate_distance(hand1, hand2)

                    if initial_distance is None:
                        # Set initial distance
                        initial_distance = current_distance
                    else:
                        # Calculate zoom scale based on distance ratio
                        zoom_scale = current_distance / initial_distance
                        # Perform zoom action
                        pyautogui.hotkey('command', '+') if zoom_scale > 1 else pyautogui.hotkey('command', '-')

            else:
                initial_distance = None

            # If one hand is detected
            if len(results.multi_hand_landmarks) == 1:
                # Get landmarks for the hand
                hand_landmarks = results.multi_hand_landmarks[0]
                myhand.draw_landmarks(image, hand_landmarks)
                # If fingers are pinching, set mouse click
                if hand.is_pinch(hand_landmarks, image) and not mouse_clicked:
                    pyautogui.mouseDown()
                    print('Mouse Down!')
                    mouse_clicked = True
                # If fingers are apart, release mouse click
                elif not hand.is_pinch(hand_landmarks, image) and mouse_clicked:
                    pyautogui.mouseUp()
                    print('Mouse Up!')
                    mouse_clicked = False

        # Display the result image
        cv.imshow('Hand Detection', image)

        # Exit if ESC key is pressed
        if cv.waitKey(1) == 27:
            break

    # Release resources
    cap.release()
    cv.destroyAllWindows()
