"""Count raised fingers on hands present in frame"""
import time
from math import degrees, sqrt

import cv2
import mediapipe as mp
import numpy as np

last_fps_check = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID)
frame_count = 0
fps = 0


def get_fps():
    """Calculate current fps"""
    global last_fps_check
    global frame_count
    global fps
    current_timestamp = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID)
    if current_timestamp - last_fps_check > 1:
        last_fps_check = current_timestamp
        fps = frame_count
        frame_count = 0


def calculate_landmarks_distance(this_lm, other_lm) -> float:
    """Calculate distance between 2 landmark co-ordinates

    Parameters
    ----------
    this_lm : NormalizedLandmark
        landmark with x,y,z components
    other_lm : NormalizedLandmark
        landmark with x,y,z components

    Returns
    -------
    float
        Scalar distance between 2 landmarks
    """
    return sqrt(
        pow(this_lm.x - other_lm.x, 2)
        + pow(this_lm.y - other_lm.y, 2)
        + pow(this_lm.z - other_lm.z, 2)
    )


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
HAND_LM = mp_hands.HandLandmark

vid = cv2.VideoCapture(0)

while True:

    ret, image = vid.read()
    frame_count += 1

    get_fps()

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        max_num_hands=2,
        min_tracking_confidence=0.5,
    ) as hands:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_height, image_width, _ = image.shape

        angle_deg = 0.0
        # List of tuple of count and position of wrists
        hand_count_and_pos: list[tuple] = []

        # Analyze each set of hand_landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                count = 0

                # Thumb
                thumb_mcp = hand_landmarks.landmark[HAND_LM.THUMB_MCP]
                thumb_ip = hand_landmarks.landmark[HAND_LM.THUMB_IP]
                thumb_tip = hand_landmarks.landmark[HAND_LM.THUMB_TIP]
                mcp_to_ip_arr = np.array(
                    [
                        thumb_ip.x - thumb_mcp.x,
                        thumb_ip.y - thumb_mcp.y,
                        thumb_ip.z - thumb_mcp.z,
                    ]
                )
                ip_to_tip_arr = np.array(
                    [
                        thumb_tip.x - thumb_ip.x,
                        thumb_tip.y - thumb_ip.y,
                        thumb_tip.z - thumb_ip.z,
                    ]
                )
                dot_prod = np.dot(mcp_to_ip_arr, ip_to_tip_arr)
                mcp_to_ip_mag = np.sqrt((mcp_to_ip_arr * mcp_to_ip_arr).sum())
                ip_to_tip_mag = np.sqrt((ip_to_tip_arr * ip_to_tip_arr).sum())
                angle_deg = degrees(
                    np.arccos((dot_prod / (mcp_to_ip_mag * ip_to_tip_mag)))
                )

                wrist_to_thumb_tip = calculate_landmarks_distance(
                    hand_landmarks.landmark[HAND_LM.WRIST],
                    hand_landmarks.landmark[HAND_LM.THUMB_TIP],
                )
                wrist_to_index_mcp = calculate_landmarks_distance(
                    hand_landmarks.landmark[HAND_LM.WRIST],
                    hand_landmarks.landmark[HAND_LM.MIDDLE_FINGER_MCP],
                )
                # Increment if thumb is bent over a certain threshold
                THUMB_ANGLE_THRESHOLD = 30
                count = (
                    count + 1
                    if (
                        wrist_to_thumb_tip > wrist_to_index_mcp
                        and angle_deg < THUMB_ANGLE_THRESHOLD
                    )
                    else count
                )

                # Index Finger
                lm_pairs = [
                    (HAND_LM.INDEX_FINGER_PIP, HAND_LM.INDEX_FINGER_TIP),
                    (HAND_LM.MIDDLE_FINGER_PIP, HAND_LM.MIDDLE_FINGER_TIP),
                    (HAND_LM.RING_FINGER_PIP, HAND_LM.RING_FINGER_TIP),
                    (HAND_LM.PINKY_PIP, HAND_LM.PINKY_TIP),
                ]

                # if finger's Proximal Phalanx is further away from tip then finger must be curled
                for pair in lm_pairs:
                    wrist_to_pip = calculate_landmarks_distance(
                        hand_landmarks.landmark[HAND_LM.WRIST],
                        hand_landmarks.landmark[pair[0]],
                    )
                    wrist_to_tip = calculate_landmarks_distance(
                        hand_landmarks.landmark[HAND_LM.WRIST],
                        hand_landmarks.landmark[pair[1]],
                    )
                    count = count + 1 if (wrist_to_tip > wrist_to_pip) else count

                pos = (
                    int(hand_landmarks.landmark[HAND_LM.WRIST].x * image_width),
                    int(hand_landmarks.landmark[HAND_LM.WRIST].y * image_height),
                )
                hand_count_and_pos.append((count, pos))

                # mp_drawing.draw_landmarks(
                #     image,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style())

        image = cv2.putText(
            image,
            f"{fps} FPS",
            (50, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
        )

        for data in hand_count_and_pos:
            image = cv2.putText(
                image,
                f"{data[0]}",
                data[1],
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=5,
                color=(0, 0, 255),
                thickness=4,
            )

        cv2.imshow("MediaPipe Hands", image)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
