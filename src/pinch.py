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
