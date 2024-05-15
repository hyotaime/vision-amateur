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
