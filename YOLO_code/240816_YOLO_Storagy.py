from ultralytics import YOLO
import cv2
import numpy as np
import logging
import time
import threading
from scipy.spatial.distance import cdist

# 상수 Define
ESC_KEY = ord('q')          # 캠 종료 버튼
WEBCAM_INDEX = 2            # 사용하고자 하는 웹캠 장치의 인덱스
FRAME_WIDTH = 640           # 웹캠 프레임 너비
FRAME_HEIGHT = 480          # 웹캠 프레임 높이
CONFIDENCE_THRESHOLD = 0.7  # YOLO 모델의 신뢰도 임계값
DEFAULT_MODEL_PATH = '/home/beakhongha/YOLO_ARIS/train23/weights/best.pt'   # YOLO 모델의 경로

CAPSULE_CHECK_ROI = [(455, 65, 95, 95), (360, 65, 95, 95), (265, 65, 95, 95)]  # A_ZONE, B_ZONE, C_ZONE 순서
SEAL_CHECK_ROI = (450, 230, 110, 110)  # Seal check ROI 구역

ROBOT_STOP_DISTANCE = 50    # 로봇이 일시정지하는 사람과 로봇 사이의 거리

logging.getLogger("ultralytics").setLevel(logging.WARNING)  # 로깅 수준을 WARNING으로 설정하여 정보 메시지 비활성화


class YOLOMain:
    def __init__(self, robot_main, model_path=DEFAULT_MODEL_PATH, webcam_index=WEBCAM_INDEX, 
                 frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT, conf=CONFIDENCE_THRESHOLD):
        """
        YOLOMain 클래스 초기화 메서드.
        모델을 로드하고 웹캠을 초기화하며, 카메라와 로봇 좌표계 간의 호모그래피 변환 행렬을 계산합니다.
        """
        self.model = YOLO(model_path)
        self.webcam = cv2.VideoCapture(webcam_index)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.conf = conf

        self.robot = robot_main

        if not self.webcam.isOpened():
            raise Exception("웹캠을 열 수 없습니다. 프로그램을 종료합니다.")
        
        # 변수 초기화
        self.center_x_mm = None
        self.center_y_mm = None

        self.init_roi_state()  # ROI 상태 초기화
        self.colors = self.init_colors()  # 객체 인식 바운딩 박스 및 마스크 색상 설정
        self.H = self.compute_homography_matrix()  # 호모그래피 변환 행렬 계산
    

    def init_roi_state(self):
        """
        ROI 상태를 초기화하는 메서드.
        """
        self.robot.A_ZONE, self.robot.B_ZONE, self.robot.C_ZONE, self.robot.NOT_SEAL = False, False, False, False
        self.robot.A_ZONE_start_time, self.robot.B_ZONE_start_time, self.robot.C_ZONE_start_time = None, None, None
        self.robot.cup_trash_detected = False
        self.robot.trash_detect_start_time = None


    def init_colors(self):
        """
        객체 인식 색상을 초기화하는 메서드.
        객체의 라벨에 따른 색상을 사전으로 반환합니다.
        """
        return {
            'cup': (0, 255, 0),
            'capsule': (0, 0, 255),
            'capsule_label': (255, 255, 0),
            'capsule_not_label': (0, 255, 255),
            'robot': (0, 165, 255),
            'human': (255, 0, 0),
            'cup_holder': (255, 255, 255)
        }


    def compute_homography_matrix(self):
        """
        호모그래피 변환 행렬을 계산하는 메서드.
        카메라 좌표와 로봇 좌표를 기반으로 호모그래피 행렬을 계산합니다.
        """
        camera_points = np.array([
            [247.0, 121.0], [306.0, 107.0], [358.0, 94.0], [238.0, 79.0], [290.0, 66.0], [342.0, 52.0]
        ], dtype=np.float32)
        
        world_points = np.array([
            [116.3, -424.9], [17.4, -456.5], [-73.2, -484.2], [140.1, -518.5], [45.6, -548.1], [-47.5, -580.8]
        ], dtype=np.float32)

        H, _ = cv2.findHomography(camera_points, world_points)
        print("호모그래피 변환 행렬 H:\n", H)
        return H
    

    def transform_to_robot_coordinates(self, image_points):
        """
        이미지 좌표를 로봇 좌표계로 변환하는 메서드.
        주어진 이미지 좌표를 로봇 좌표계로 변환합니다.

        :param image_points: 이미지 좌표 [x, y]
        :return: 로봇 좌표계로 변환된 좌표 [x, y]
        """
        camera_coords = np.array([[image_points]], dtype=np.float32)
        robot_coords = cv2.perspectiveTransform(camera_coords, self.H)
        return [round(float(coord), 1) for coord in robot_coords[0][0]]


    def predict_on_image(self, img):
        """
        입력된 이미지에 대해 예측을 수행하는 메서드.
        YOLO 모델을 사용해 바운딩 박스, 마스크, 클래스, 신뢰도 점수를 반환합니다.

        :param img: 예측할 이미지
        :return: 바운딩 박스, 마스크, 클래스, 신뢰도 점수
        """
        result = self.model(img, conf=self.conf)[0]

        cls = result.boxes.cls.cpu().numpy() if result.boxes else []
        probs = result.boxes.conf.cpu().numpy() if result.boxes else []
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        
        return boxes, masks, cls, probs


    def overlay(self, image, mask, color, alpha=0.5):
        """
        이미지 위에 세그멘테이션 마스크를 오버레이하는 메서드.
        주어진 색상과 투명도를 사용하여 마스크를 원본 이미지에 결합합니다.

        :param image: 원본 이미지
        :param mask: 세그멘테이션 마스크
        :param color: 마스크를 표시할 색상
        :param alpha: 마스크와 원본 이미지의 혼합 비율
        :return: 마스크가 오버레이된 이미지
        """
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]
        
        try:
            mask_indices = mask > 0
            overlay_image = image.copy()
            overlay_image[mask_indices] = cv2.addWeighted(image[mask_indices], 1 - alpha, colored_mask[mask_indices], alpha, 0)
        except Exception as e:
            print(f"오버레이 처리 중 오류 발생: {e}")
            return image  # 오류 발생 시 원본 이미지를 반환
        
        return overlay_image
    

    def find_contours(self, mask):
        """
        마스크에서 외곽선을 찾는 메서드.
        """
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    

    def pause_robot(self, image_with_masks, robot_contours, human_contours):
        """
        로봇과 인간 간의 최단 거리를 계산하고 로봇을 일시정지하게 하는 메서드.
        """
        # 사람과 로봇 사이의 최단 거리 계산 및 시각화
        if robot_contours and human_contours:
            robot_points = np.vstack(robot_contours).squeeze()
            human_points = np.vstack(human_contours).squeeze()
            dists = cdist(robot_points, human_points)
            min_dist_idx = np.unravel_index(np.argmin(dists), dists.shape)
            robot_point = robot_points[min_dist_idx[0]]
            human_point = human_points[min_dist_idx[1]]
            min_distance = dists[min_dist_idx]
            min_distance_bool = True

            # 사람과 로봇 사이의 최단 거리 표시
            cv2.line(image_with_masks, tuple(robot_point), tuple(human_point), (255, 255, 255), 2)
            mid_point = ((robot_point[0] + human_point[0]) // 2, (robot_point[1] + human_point[1]) // 2)
            cv2.putText(image_with_masks, f'{min_distance:.2f}', mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            min_distance = 300
            min_distance_bool = False

        # 거리 조건 체크 및 로봇 일시정지 제어
        if min_distance <= ROBOT_STOP_DISTANCE and min_distance_bool and self.robot.pressing == False:
            self.robot.robot_state = 'robot stop'
            # self.robot._arm.set_state(3)
        elif min_distance > ROBOT_STOP_DISTANCE or not min_distance_bool:
            self.robot.robot_state = 'robot move'
            # self.robot._arm.set_state(0)

        # 화면 왼쪽 위에 최단 거리 및 로봇 상태 및 ROI 상태 표시
        cv2.putText(image_with_masks, f'Distance: {min_distance:.2f}, state: {self.robot.robot_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image_with_masks, f'A_ZONE: {self.robot.A_ZONE}, B_ZONE: {self.robot.B_ZONE}, C_ZONE: {self.robot.C_ZONE}, NOT_SEAL: {self.robot.NOT_SEAL}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    def capsule_detect_check(self, x1, y1, x2, y2, roi, zone_name, zone_flag, start_time):
        """
        ROI 영역에서 객체가 일정 시간 이상 감지되었는지 확인하는 메서드.
        """
        rx, ry, rw, rh = roi
        intersection_x1 = max(x1, rx)
        intersection_y1 = max(y1, ry)
        intersection_x2 = min(x2, rx + rw)
        intersection_y2 = min(y2, ry + rh)
        intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
        box_area = (x2 - x1) * (y2 - y1)
        is_condition_met = intersection_area >= 0.8 * box_area

        if is_condition_met:
            current_time = time.time()
            if not zone_flag:
                if start_time is None:
                    start_time = current_time
                    print(f'{zone_name} start time set')
                elif current_time - start_time >= 2:
                    zone_flag = True
                else:
                    print(f'Waiting for 2 seconds: {current_time - start_time:.2f} seconds elapsed')
            else:
                start_time = current_time
        else:
            start_time = None

        return zone_flag, start_time
    

    def seal_remove_check(self, x1, y1, x2, y2, roi, zone_flag):
        """
        ROI 영역에서 객체가 감지되었는지 확인하는 메서드.
        """
        rx, ry, rw, rh = roi
        intersection_x1 = max(x1, rx)
        intersection_y1 = max(y1, ry)
        intersection_x2 = min(x2, rx + rw)
        intersection_y2 = min(y2, ry + rh)
        intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
        box_area = (x2 - x1) * (y2 - y1)
        
        if intersection_area >= 0.8 * box_area:
            zone_flag = True

        return zone_flag


    # 객체의 현재 위치와 과거 위치의 차이를 비교하기 위한 함수
    def distance_between_points(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


    def run_yolo(self):
        """
        YOLO 모델을 실행하는 메서드.
        실시간으로 웹캠 영상을 처리하고, 예측 결과를 화면에 출력합니다.
        """
        # 카메라 작동
        while True:
            ret, frame = self.webcam.read()  # 웹캠에서 프레임 읽기
            if not ret:  # 프레임을 읽지 못한 경우
                print("카메라에서 프레임을 읽을 수 없습니다. 프로그램을 종료합니다.")  # 오류 메시지 출력
                break

            # 현재 프레임 예측
            boxes, masks, cls, probs = self.predict_on_image(frame)

            # 원본 이미지에 마스크 오버레이 및 디텍션 박스 표시
            image_with_masks = np.copy(frame)  # 원본 이미지 복사

            robot_contours = []
            human_contours = []

            # 설정된 ROI를 흰색 바운딩 박스로 그리고 선을 얇게 설정
            for (x, y, w, h) in CAPSULE_CHECK_ROI:
                cv2.rectangle(image_with_masks, (x, y), (x + w, y + h), (255, 255, 255), 1)  # 각 ROI를 흰색 사각형으로 그림
            # 특정 ROI를 흰색 바운딩 박스로 그리고 선을 얇게 설정
            cv2.rectangle(image_with_masks, (SEAL_CHECK_ROI[0], SEAL_CHECK_ROI[1]), 
                          (SEAL_CHECK_ROI[0] + SEAL_CHECK_ROI[2], SEAL_CHECK_ROI[1] + SEAL_CHECK_ROI[3]), 
                          (255, 255, 255), 1)  # 특정 ROI를 흰색 사각형으로 그림
            
            # 각 객체에 대해 박스, 마스크 생성
            for box, mask, class_id, prob in zip(boxes, masks, cls, probs):  # 각 객체에 대해
                label = self.model.names[int(class_id)]  # 클래스 라벨 가져오기

                if label == 'hand':  # 'hand' 객체를 'human' 객체로 변경
                    label = 'human'

                color = self.colors.get(label, (255, 255, 255))  # 클래스에 해당하는 색상 가져오기
                
                if mask is not None and len(mask) > 0:
                    # 마스크 오버레이
                    image_with_masks = self.overlay(image_with_masks, mask, color, alpha=0.3)

                    # 라벨별 외곽선 저장
                    contours = self.find_contours(mask)
                    if label == 'robot':
                        robot_contours.extend(contours)
                    elif label == 'human':
                        human_contours.extend(contours)

                # 디텍션 박스 및 라벨 표시
                x1, y1, x2, y2 = map(int, box)  # 박스 좌표 정수형으로 변환
                cv2.rectangle(image_with_masks, (x1, y1), (x2, y2), color, 2)  # 경계 상자 그리기                        
                cv2.putText(image_with_masks, f'{label} {prob:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 라벨 및 신뢰도 점수 표시

                # A_ZONE, B_ZONE, C_ZONE ROI 내 일정 시간 이상 capsule 객체 인식 확인
                if label == 'capsule':
                    self.robot.A_ZONE, self.robot.A_ZONE_start_time = self.capsule_detect_check(x1, y1, x2, y2, CAPSULE_CHECK_ROI[0], 'A_ZONE', self.robot.A_ZONE, self.robot.A_ZONE_start_time)
                    self.robot.B_ZONE, self.robot.B_ZONE_start_time = self.capsule_detect_check(x1, y1, x2, y2, CAPSULE_CHECK_ROI[1], 'B_ZONE', self.robot.B_ZONE, self.robot.B_ZONE_start_time)
                    self.robot.C_ZONE, self.robot.C_ZONE_start_time = self.capsule_detect_check(x1, y1, x2, y2, CAPSULE_CHECK_ROI[2], 'C_ZONE', self.robot.C_ZONE, self.robot.C_ZONE_start_time)

                # 씰 확인 ROI 내 capsule_not_label 객체 인식 확인
                if label == 'capsule_not_label':
                    self.robot.NOT_SEAL = self.seal_remove_check(x1, y1, x2, y2, SEAL_CHECK_ROI, self.robot.NOT_SEAL)

            # 로봇 일시정지 기능
            self.pause_robot(image_with_masks, robot_contours, human_contours)
            
            # 디텍션 박스와 마스크가 적용된 프레임 표시
            cv2.imshow("Webcam with Segmentation Masks and Detection Boxes", image_with_masks)

            # 종료 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ESC_KEY:
                break

        # 자원 해제
        self.webcam.release()  # 웹캠 장치 해제
        cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기



class RobotMain():
    def __init__(self, **kwargs):
        self.A_ZONE, self.B_ZONE, self.C_ZONE, self.NOT_SEAL = False, False, False, False
        self.A_ZONE_start_time, self.B_ZONE_start_time, self.C_ZONE_start_time = None, None, None
        self.cup_trash_detected = False
        self.trash_detect_start_time = None
        self.pressing = False

    def run_robot():
        '''
        무한루프
        '''
        while(1):
            i = 1



if __name__ == "__main__":
    robot_main = RobotMain()
    yolo_main = YOLOMain(robot_main)

    robot_thread = threading.Thread(target=robot_main.run_robot)
    yolo_thread = threading.Thread(target=yolo_main.run_yolo)

    robot_thread.start()
    yolo_thread.start()