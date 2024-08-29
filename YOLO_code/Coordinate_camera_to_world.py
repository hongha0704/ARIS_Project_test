import numpy as np
import cv2 as cv

# 저장된 캘리브레이션 데이터 로드
calibration_data = np.load('camera_calibration/calibration_data.npz')
mtx = calibration_data['mtx']
dist = calibration_data['dist']

# 이미지 좌표를 실세계 좌표로 변환하는 함수 정의
def get_object_coordinates(image_points, mtx, dist):
    # 왜곡을 보정하여 이미지 좌표를 변환
    object_points = cv.undistortPoints(np.expand_dims(image_points, axis=1), mtx, dist)

    # 변환된 좌표를 동차 좌표계로 변환 (homogeneous coordinates)
    object_points_3D = cv.convertPointsToHomogeneous(object_points)
    object_points_3D[:, :, 2] = 0  # Z 값을 0으로 설정하여 평면 상의 좌표로 변환

    return object_points_3D

# 웹캠에서 실시간으로 물체 좌표 변환
cap = cv.VideoCapture(2)  # 웹캠 장치 열기 (장치 번호 2번 사용)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 이미지 좌표를 예시로 설정 (실제로는 물체 검출 알고리즘 필요)
    # 예시로 이미지를 클릭하여 좌표를 얻는다고 가정
    # 이 예제에서는 (500, 300) 좌표를 사용합니다.
    image_points = np.array([[500, 300]], dtype=np.float32)

    # 실세계 좌표 계산
    object_points_3D = get_object_coordinates(image_points, mtx, dist)

    # 결과 출력
    print("실세계 좌표:", object_points_3D)

    # 웹캠 프레임에 좌표 표시
    cv.circle(frame, (int(image_points[0][0]), int(image_points[0][1])), 5, (0, 255, 0), -1)  # 좌표에 원 표시
    cv.putText(frame, f"World Coord: {object_points_3D[0][0]}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # 좌표 텍스트 표시
    
    cv.imshow('Webcam', frame)  # 프레임 보여주기

    # 'q' 키를 누르면 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 웹캠 릴리즈
cv.destroyAllWindows()  # 모든 창 닫기
