def get_hand_avg_pos(hand_landmarks, frame_shape):
    """
    Calculate the average position of the hand tips (thumb, index, middle, ring, pinky) in image coordinates.

    Args:
    hand_landmarks (object): The detected hand landmarks.
    frame_shape (tuple): The shape of the frame (height, width, channels).

    Returns:
    tuple: The average x, y coordinates of the hand tips in image coordinates.
    """
    # Indices for hand tip landmarks
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    # Get the normalized coordinates for each fingertip
    thumb_tip = hand_landmarks.landmark[THUMB_TIP]
    index_tip = hand_landmarks.landmark[INDEX_TIP]
    middle_tip = hand_landmarks.landmark[MIDDLE_TIP]
    ring_tip = hand_landmarks.landmark[RING_TIP]
    pinky_tip = hand_landmarks.landmark[PINKY_TIP]

    # Calculate the average normalized coordinates
    avg_x = (thumb_tip.x + index_tip.x + middle_tip.x + ring_tip.x + pinky_tip.x) / 5
    avg_y = (thumb_tip.y + index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y) / 5

    # Convert normalized coordinates to image coordinates
    return int(avg_x * frame_shape[1]), int(avg_y * frame_shape[0])

def calculate_distance(hand1, hand2):
    """
    Calculate the distance between the index finger tips of two hands.

    Args:
    hand1 (object): The first detected hand landmarks.
    hand2 (object): The second detected hand landmarks.

    Returns:
    float: The Euclidean distance between the index fingertips of the two hands.
    """
    INDEX_TIP = 8  # Index for the index fingertip landmark

    # Get the normalized coordinates for the index fingertips
    hand1_index_tip = hand1.landmark[INDEX_TIP]
    hand2_index_tip = hand2.landmark[INDEX_TIP]

    # Calculate the differences in x and y coordinates
    dx = hand1_index_tip.x - hand2_index_tip.x
    dy = hand1_index_tip.y - hand2_index_tip.y

    # Calculate and return the Euclidean distance
    return (dx ** 2 + dy ** 2) ** 0.5
