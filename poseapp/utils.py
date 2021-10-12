import cv2
import math
import numpy as np


def output_keypoints(frame, proto_file, weights_file, threshold, model_name, BODY_PARTS):
    global points
    global count
    # 이미지 읽어오기
    # frame = cv2.imread(image_path)
    # cv2.im

    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # 입력 이미지의 사이즈 정의
    image_height = 368
    image_width = 368

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False,
                                       crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward()

    out_height = out.shape[2]

    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    points = []

    print(f"\n========== {model_name} ==========")
    count = [0, 0]
    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정
        x = (frame_width * point[0]) / out_width
        x = int(x)
        y = (frame_height * point[1]) / out_height
        y = int(y)
        if prob > threshold:  # [pointed]
            if i in (0, 2, 5, 17, 18):
                cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                if i in (2, 17):
                    count[1] = count[1] + 1
                if i in (5, 18):
                    count[0] = count[0] + 1

            points.append((x, y))
            # print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

        else:  # [not pointed]
            # if i in (0, 2, 5, 17, 18):
            #     cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            #     cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            points.append(None)
            # print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    return frame


def output_keypoints_with_lines(POSE_PAIRS, frame):
    # 프레임 복사
    frame_line = frame.copy()

    # Neck 과 MidHeap 의 좌표값이 존재한다면
    if count[0] > count[1]:
        print("left")
        # calculate_degree(point_1=points[5], point_2=points[18], point_3=points[5], point_4=points[0], frame=frame_line)
        calculate_degree(point_1=points[5], point_2=points[18], frame=frame_line)
    elif count[0] < count[1]:
        print("right")
        # calculate_degree(point_1=points[2], point_2=points[17], point_3=points[2], point_4=points[0], frame=frame_line)
        calculate_degree(point_1=points[2], point_2=points[17], frame=frame_line)
    # if (points[5] is not None) and (points[18] is not None) and (points[2] is not None):

    for pair in POSE_PAIRS:
        part_a = pair[0]  # 0 (Head)
        part_b = pair[1]  # 1 (Neck)
        if points[part_a] and points[part_b]:
            print(f"[linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
            # Neck 과 MidHip 이라면 분홍색 선
            # if (part_a == 5 and part_b == 18) or (part_a == 18 and part_b == 5) or (part_a == 1 and part_b == 8):
            #     cv2.line(frame, points[part_a], points[part_b], (255, 0, 255), 3)
            # else:  # 노란색 선
            cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 3)
        else:
            print(f"[not linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")

    # 포인팅 되어있는 프레임과 라인까지 연결된 프레임을 가로로 연결
    frame_horizontal = cv2.hconcat([frame, frame_line])
    return frame_horizontal


def calculate_degree(point_1, point_2, frame):
    # 역탄젠트 구하기
    dx1 = abs(point_2[0] - point_1[0])
    dy1 = abs(point_2[1] - point_1[1])
    # dx2 = abs(point_4[0] - point_3[0])
    # dy2 = abs(point_4[1] - point_3[1])
    rad1 = math.atan2(abs(dy1), abs(dx1))
    # rad2 = math.atan2(abs(dy2), abs(dx2))

    # radian 을 degree 로 변환
    deg1 = rad1 * 180 / math.pi
    # deg2 = rad2 * 180 / math.pi

    # if point_1[1]>
    deg = deg1 # - deg2
    # degree 가 30'보다 작으면 거북목이라 판단
    if deg > 75:
        string = f"not {deg: .0f}"#, {deg1: .0f}, {deg2: .0f}"
        cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255))
        print(f"[degree] {deg} ({string})")
    else:
        string = f"turtle {deg: .0f}"#, {deg1: .0f}, {deg2: .0f}"
        cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255))
        print(f"[degree] {deg} ({string})")


BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                      10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                      15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                      20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel",
                      25: "Background"}

POSE_PAIRS_BODY_25 = [[5, 18], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13],
                      [2, 3],
                      [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21],
                      [20, 21],
                      [11, 24], [22, 24], [23, 24]]





