# Vision Amateur

Provide Intuitive Gestures for Computer Control

## Introduction
`vision-amateur` is a Python project that uses computer vision to detect facial landmarks and hand gestures to control your computer.
Last year, Apple showcased the Vision Pro with impressive gestures that are easy and intuitive to use.
Inspired by this concept, the project aims to provide gestures that make it easier to control the computer, 
which is why it is named "**Vision Amateur**."
It utilizes libraries such as OpenCV, PyAutoGUI, and MediaPipe to implement features 
like blink detection, mouse control via eye movements, and zooming through hand gestures.

## Features

- **Eye Blink Detection** 
  - Detects eye blinks using facial landmarks and performs a mouse click event.
- **Mouse Control via Eye Movements** 
  - Moves the mouse cursor based on the position of the eyes.
- **Hand Gesture Recognition** 
  - Recognizes hand gestures to perform actions like clicking and zooming.
- **Zoom Control** 
  - Uses the distance between two hands to control zoom in and out.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.11+ installed on your machine.
  - I tested it only on Python 3.11.9, but it might work on other versions like 3.8+.
- A webcam connected to your computer.
- The following Python libraries:
  - OpenCV (`cv2`)
  - PyAutoGUI
  - MediaPipe

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/hyotaime/vision-amateur.git
    ```

2. Navigate to the project directory:

    ```sh
    cd vision-amateur
    ```

3. Install the required libraries:

    ```sh
    pip install opencv-python pyautogui mediapipe
    ```
   or
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the main script:

    ```sh
    python main.py
    ```

2. The application will open your default webcam and start detecting facial landmarks and hand gestures.

3. To exit the application, press the `ESC` key.

## Code Overview

### main.py

The main script initializes the face and hand detection modules and processes video frames from the webcam to detect and respond to facial landmarks and hand gestures.

#### Key Components

- **Face Detection:**
  - Detects facial landmarks.
  - Calculates the Eye Aspect Ratio (EAR) to detect blinks.
  - Moves the mouse cursor based on eye position.

- **Hand Detection:**
  - Detects hand landmarks.
  - Recognizes pinching gestures for mouse click events.
  - Measures the distance between hands to control zoom.

#### Constants

- `LEFT_EYE_LANDMARKS` and `RIGHT_EYE_LANDMARKS`: Indices for the eye landmarks.
- `EYE_AR_THRESH`: Threshold for detecting eye blinks.
- `mouse_clicked` and `initial_distance`: Variables for managing mouse click state and zoom functionality.

### Helper Modules

- **common:** Contains utility functions like distance calculation.
- **hand:** Manages hand detection and gesture recognition.
- **face:** Manages face detection and processing of facial landmarks.

## Limitations
- It doesn't recognize blinking well unless looking directly at the camera.
- Dragging sometimes gets interrupted midway.
- Zoom function can only be used in a web browser.
- Zoom function is implemented with hotkeys, so precise control of the zoom level is not possible.
- During testing, keeping my eyes wide open caused dryness, leading to continuous tearing.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit them (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## References

- PyAutoGUI
  - [asweigart/pyautogui](https://github.com/asweigart/pyautogui)
- MediaPipe
  - [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe)
- Special thanks to ChatGPT-4 
  - Assisting me with troubleshooting, annotating, and also improving my poor English.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
