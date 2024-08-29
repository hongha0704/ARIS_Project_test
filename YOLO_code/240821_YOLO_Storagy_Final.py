from ultralytics import YOLO
import cv2
import numpy as np
import logging
import time
import threading
from scipy.spatial.distance import cdist


'''상수 Define'''
ESC_KEY = ord('q')           # 캠 종료 버튼
WEBCAM_INDEX = 2             # 사용하고자 하는 웹캠 장치의 인덱스
FRAME_WIDTH = 640            # 웹캠 프레임 너비
FRAME_HEIGHT = 480           # 웹캠 프레임 높이
CONFIDENCE_THRESHOLD = 0.85  # YOLO 모델의 신뢰도 임계값
DEFAULT_MODEL_PATH = '/home/beakhongha/YOLO_ARIS/train25/weights/best.pt'   # YOLO 모델의 경로

CAPSULE_CHECK_ROI = [(460, 190, 90, 90), (370, 190, 90, 90), (280, 190, 90, 90)]  # A_ZONE, B_ZONE, C_ZONE 순서
SEAL_CHECK_ROI = (475, 360, 110, 110)   # Seal check ROI 구역
CUP_TRASH_ROI = (100, 20, 520, 210)     # storagy 위의 컵 쓰레기 인식 ROI 구역

ROBOT_STOP_DISTANCE = 50            # 로봇이 일시정지하는 사람과 로봇 사이의 거리
CAPSULE_DETECTION_AREA_RATIO = 0.8  # 캡슐을 객체 인식하는 면적 비율

CAPSULE_DETECTION_TIME = 2  # 캡슐 인식 시간
CUP_DETECTION_TIME = 1      # 컵 인식 시간

DISTANCE_BETWEEN_POINTS = 10    # 중심좌표가 일정 거리 이하로 변동 시 중심좌표의 변동이 없다고 판단

logging.getLogger("ultralytics").setLevel(logging.WARNING)  # 로깅 수준을 WARNING으로 설정하여 정보 메시지 비활성화



class YOLOMain:
    def __init__(self, robot_main, model_path=DEFAULT_MODEL_PATH, webcam_index=WEBCAM_INDEX, 
                 frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT, conf=CONFIDENCE_THRESHOLD):
        """
        YOLOMain 클래스 초기화 메서드
        모델을 로드하고 웹캠을 초기화하며, 카메라와 로봇 좌표계 간의 호모그래피 변환 행렬을 계산
        """
        # 모델 로드, 웹캠 초기화
        self.model = YOLO(model_path)
        self.webcam = cv2.VideoCapture(webcam_index)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.conf = conf

        self.robot = robot_main

        if not self.webcam.isOpened():
            raise Exception("웹캠을 열 수 없습니다. 프로그램을 종료합니다.")
        
        # 컵, 컵홀더 좌표 초기화
        self.cup_trash_x, self.cup_trash_y = None, None
        self.cup_trash_x_pixel, self.cup_trash_y_pixel = None, None
        self.last_cup_trash_center = None

        self.cup_holder_x, self.cup_holder_y = None, None
        self.cup_holder_x_pixel, self.cup_holder_y_pixel = None, None
        self.last_cup_holder_center = None

        # ROI 상태 초기화
        self.init_roi_state()

        # 객체 인식 바운딩 박스 및 마스크 색상 설정
        self.colors = self.init_colors()

        # 호모그래피 변환 행렬 계산         
        self.homography_matrix = self.compute_homography_matrix()
    

    def init_roi_state(self):
        """
        ROI 상태를 초기화하는 메서드
        """
        # 캡슐, 씰 제거 여부 확인 변수 초기화
        self.robot.A_ZONE, self.robot.B_ZONE, self.robot.C_ZONE, self.robot.NOT_SEAL = False, False, False, False
        self.robot.A_ZONE_start_time, self.robot.B_ZONE_start_time, self.robot.C_ZONE_start_time = None, None, None

        # 컵, 컵홀더 탐지 변수 초기화
        self.robot.cup_trash_detected, self.robot.cup_holder_detected = False, False
        self.robot.cup_trash_detect_start_time, self.robot.cup_holder_detect_start_time = None, None


    def init_colors(self):
        """
        객체 인식 색상을 초기화하는 메서드
        객체의 라벨에 따른 색상을 사전으로 반환
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
        호모그래피 변환 행렬을 계산하는 메서드
        카메라 좌표와 로봇 좌표를 기반으로 호모그래피 행렬을 계산
        """
        # 카메라 좌표, 로봇 좌표
        camera_points = np.array([
            [247.0, 121.0], [306.0, 107.0], [358.0, 94.0], [238.0, 79.0], [290.0, 66.0], [342.0, 52.0]
        ], dtype=np.float32)
        
        robot_points = np.array([
            [116.3, -424.9], [17.4, -456.5], [-73.2, -484.2], [140.1, -518.5], [45.6, -548.1], [-47.5, -580.8]
        ], dtype=np.float32)

        # 변환 행렬 계산
        homography_matrix, _ = cv2.findHomography(camera_points, robot_points)
        print("호모그래피 변환 행렬 homography_matrix:\n", homography_matrix)

        return homography_matrix
    

    def transform_to_robot_coordinates(self, image_points):
        """
        이미지 좌표를 로봇 좌표계로 변환하는 메서드
        주어진 이미지 좌표를 로봇 좌표계로 변환
        """
        camera_coords = np.array([[image_points]], dtype=np.float32)
        robot_coords = cv2.perspectiveTransform(camera_coords, self.homography_matrix)

        return [round(float(coord), 1) for coord in robot_coords[0][0]]


    def predict_on_image(self, img):
        """
        입력된 이미지에 대해 예측을 수행하는 메서드
        YOLO 모델을 사용해 바운딩 박스, 마스크, 클래스, 신뢰도 점수를 반환
        """
        result = self.model(img, conf=self.conf)[0]

        cls = result.boxes.cls.cpu().numpy() if result.boxes else []
        probs = result.boxes.conf.cpu().numpy() if result.boxes else []
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        
        # 예측 결과 반환(박스, 마스크, 클래스, 신뢰도 점수)
        return boxes, masks, cls, probs


    def overlay(self, image, mask, color, alpha=0.5):
        """
        이미지 위에 세그멘테이션 마스크를 오버레이하는 메서드
        주어진 색상과 투명도를 사용하여 마스크를 원본 이미지에 결합
        """
        # 마스크 크기를 조정하고 색상으로 칠함
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]
        
        # 오버레이 이미지를 생성하고 반환
        try:
            mask_indices = mask > 0
            overlay_image = image.copy()
            overlay_image[mask_indices] = cv2.addWeighted(image[mask_indices], 1 - alpha, colored_mask[mask_indices], alpha, 0)
            return overlay_image
        
        # 오류 발생 시 원본 이미지를 반환
        except Exception as e:
            print(f"오버레이 처리 중 오류 발생: {e}")
            return image  
        

    def find_contours(self, mask):
        """
        마스크에서 외곽선을 찾는 메서드
        """
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    

    def pause_robot(self, image_with_masks, robot_contours, human_contours):
        """
        로봇과 인간 간의 최단 거리를 계산하고 로봇을 일시정지하게 하는 메서드
        """
        # 사람과 로봇 사이의 최단 거리 계산
        if robot_contours and human_contours:
            robot_points = np.vstack(robot_contours).squeeze()
            human_points = np.vstack(human_contours).squeeze()
            dists = cdist(robot_points, human_points)
            min_dist_idx = np.unravel_index(np.argmin(dists), dists.shape)
            robot_point = robot_points[min_dist_idx[0]]
            human_point = human_points[min_dist_idx[1]]
            self.min_distance = dists[min_dist_idx]
            min_distance_bool = True

            # 사람과 로봇 사이의 최단 거리 표시
            cv2.line(image_with_masks, tuple(robot_point), tuple(human_point), (255, 255, 255), 2)
            mid_point = ((robot_point[0] + human_point[0]) // 2, (robot_point[1] + human_point[1]) // 2)
            cv2.putText(image_with_masks, f'{self.min_distance:.2f}', mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 사람 또는 로봇의 외곽선 없을 때 최단 거리 비활성화
        else:
            self.min_distance = 300
            min_distance_bool = False

        # 거리 조건 체크 및 로봇 일시정지 제어
        if self.min_distance <= ROBOT_STOP_DISTANCE and min_distance_bool and self.robot.pressing == False:
            self.robot.robot_state = 'robot stop'
            # self.robot._arm.set_state(3)
        elif self.min_distance > ROBOT_STOP_DISTANCE or not min_distance_bool:
            self.robot.robot_state = 'robot move'
            # self.robot._arm.set_state(0)


    def capsule_detect_check(self, x1, y1, x2, y2, roi, zone_name, zone_flag, start_time):
        """
        ROI 영역에서 객체가 일정 시간 이상 감지되었는지 확인하는 메서드
        """
        # ROI와 바운딩 박스의 교차 영역 계산
        rx, ry, rw, rh = roi
        intersection_x1 = max(x1, rx)
        intersection_y1 = max(y1, ry)
        intersection_x2 = min(x2, rx + rw)
        intersection_y2 = min(y2, ry + rh)
        intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
        box_area = (x2 - x1) * (y2 - y1)

        # 교차 영역이 바운딩 박스 면적의 일정 비율 이상인지 여부
        if intersection_area >= CAPSULE_DETECTION_AREA_RATIO * box_area:
            is_condition_met = True
        else:
            is_condition_met = False

        # ROI 내에서 capsule 객체 일정 시간 이상 인식 확인
        if is_condition_met:
            current_time = time.time()
            if not zone_flag:
                if start_time is None:
                    start_time = current_time
                    print(f'{zone_name} start time set')
                elif current_time - start_time >= CAPSULE_DETECTION_TIME:
                    zone_flag = True
                else:
                    print(f'Waiting for {CAPSULE_DETECTION_TIME} seconds: {current_time - start_time:.2f} seconds elapsed')
            else:
                start_time = current_time
        else:
            start_time = None

        # ROI 상태 및 인식 시작 시간 반환
        return zone_flag, start_time
    

    def seal_remove_check(self, x1, y1, x2, y2, roi, zone_flag):
        """
        ROI 영역에서 객체가 감지되었는지 확인하는 메서드
        """
        # 씰 제거 여부 확인 ROI와 바운딩 박스의 교차 영역 계산
        rx, ry, rw, rh = roi
        intersection_x1 = max(x1, rx)
        intersection_y1 = max(y1, ry)
        intersection_x2 = min(x2, rx + rw)
        intersection_y2 = min(y2, ry + rh)
        intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
        box_area = (x2 - x1) * (y2 - y1)
        
        # 교차 영역이 바운딩 박스 면적의 일정 비율 이상인지 여부
        if intersection_area >= CAPSULE_DETECTION_AREA_RATIO * box_area:
            zone_flag = True

        # ROI 상태 반환
        return zone_flag
    

    def make_object_list(self, x1, y1, x2, y2, image_with_masks, object_list, object_list_pixel):
        '''
        ROI 영역에서 객체(컵, 컵 홀더)가 감지되었는지 확인하고 리스트에 중심 좌표를 저장하는 메서드
        '''
        # center 좌표(pixel)
        center_x_pixel = (x2 - x1) / 2 + x1
        center_y_pixel = (y2 - y1) / 2 + y1

        # ROI 영역 내에 있는지 확인
        if CUP_TRASH_ROI[0] <= center_x_pixel <= CUP_TRASH_ROI[2] and CUP_TRASH_ROI[1] <= center_y_pixel <= CUP_TRASH_ROI[3]:
            # 이미지 좌표로 실세계 좌표 계산
            image_points = [center_x_pixel, center_y_pixel]
            world_points = self.transform_to_robot_coordinates(image_points)
            center_x_mm, center_y_mm = world_points
            
            # 리스트에 좌표값 추가
            object_list_pixel.append((center_x_pixel, center_y_pixel))
            object_list.append((center_x_mm, center_y_mm))
            
            # 중심좌표 화면에 출력
            cv2.putText(image_with_masks, f'Center: ({int(center_x_mm)}, {int(center_y_mm)})', (int(center_x_pixel), int(center_y_pixel - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(image_with_masks, (int(center_x_pixel), int(center_y_pixel)), 5, (255, 0, 0), -1)

        # 중심 좌표 저장된 리스트 반환
        return object_list, object_list_pixel


    def object_detect_order(self, image_with_masks, zone_flag, start_time, set_object_coordinates, last_object_center,
                               object_x, object_y, object_max_y, object_list,
                               object_x_pixel, object_y_pixel, object_max_y_pixel, object_list_pixel):
        '''
        ARIS에서 가장 가까이 있는 객체(컵, 컵 홀더)의 좌표값을 받아오고, 일정 시간 이상 좌표값의 변동이 없는지 확인하는 메서드
        '''
        # 가장 큰 y 좌표를 가진 객체를 찾음
        for x, y in object_list_pixel:
            if y > object_max_y_pixel:
                object_max_y_pixel = y
                object_x_pixel = x
                object_y_pixel = y

        for x, y in object_list:
            if y > object_max_y:
                object_max_y = y
                object_x = x
                object_y = y

        # 좌표 정보를 로봇에 전송
        set_object_coordinates(object_x, object_y)

        # 중심좌표 중에 ARIS와 가장 가까운 값 다른 색으로 화면에 출력
        cv2.putText(image_with_masks, f'Center: ({int(object_x)}, {int(object_y)})', (int(object_x_pixel), int(object_y_pixel - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(image_with_masks, (int(object_x_pixel), int(object_y_pixel)), 5, (0, 0, 255), -1)

        # 일정 시간 이상 중심좌표의 변동 없이 감지되는지 확인
        if last_object_center:
            if self.distance_between_points((object_x, object_y), last_object_center) < DISTANCE_BETWEEN_POINTS:
                current_time = time.time()
                if start_time is None:
                    start_time = current_time
                    print('object detect start time set')
                elif current_time - start_time >= CUP_DETECTION_TIME:
                    zone_flag = True
                else:
                    print(f"object detected for {current_time - start_time:.2f} seconds")
            else:
                start_time = None
                zone_flag = False
        # 중심좌표 갱신
        last_object_center = (object_x, object_y)

        # ROI 상태, 인식 시작 시간, 갱신한 중심좌표 반환
        return zone_flag, start_time, last_object_center


    def distance_between_points(self, p1, p2):
        """
        객체의 현재 위치와 과거 위치의 차이를 비교하기 위한 메서드
        """
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


    def run_yolo(self):
        """
        YOLO 모델을 실행하는 메서드
        실시간으로 웹캠 영상을 처리 및 예측 결과를 화면에 출력하고, 여러 기능을 실행
        """
        # 카메라 작동
        while True:
            # 웹캠에서 프레임 읽기
            ret, frame = self.webcam.read()

            # 프레임을 읽지 못한 경우 오류 메시지 출력, 프로그램 종료
            if not ret: 
                print("카메라에서 프레임을 읽을 수 없습니다. 프로그램을 종료합니다.")
                break

            # 현재 프레임 예측
            boxes, masks, cls, probs = self.predict_on_image(frame)

            # 원본 이미지에 마스크 오버레이 및 디텍션 박스 표시
            image_with_masks = np.copy(frame)

            # 사람과 로봇의 segmentation 마스크 외곽선을 저장하는 리스트 (프레임 마다 초기화)
            robot_contours = []
            human_contours = []

            # ROI 영역 내 객체(컵, 컵홀더) 좌표를 저장하는 리스트 (프레임 마다 초기화)
            self.cup_trash_list = []
            self.cup_trash_list_pixel = []
            self.cup_holder_list = []
            self.cup_holder_list_pixel = []

            # 객체(컵, 컵홀더) y좌표 비교용 변수 (프레임 마다 초기화)
            self.cup_trash_max_y = -float('inf')
            self.cup_trash_max_y_pixel = -float('inf')
            self.cup_holder_max_y = -float('inf')
            self.cup_holder_max_y_pixel = -float('inf')

            # 캡슐을 인식하는 ROI를 흰색 바운딩 박스로 그리고 선을 얇게 설정
            for (x, y, w, h) in CAPSULE_CHECK_ROI:
                cv2.rectangle(image_with_masks, (x, y), (x + w, y + h), (255, 255, 255), 1)

            # 씰 제거 여부 확인 ROI를 흰색 바운딩 박스로 그리고 선을 얇게 설정
            cv2.rectangle(image_with_masks, (SEAL_CHECK_ROI[0], SEAL_CHECK_ROI[1]), 
                          (SEAL_CHECK_ROI[0] + SEAL_CHECK_ROI[2], SEAL_CHECK_ROI[1] + SEAL_CHECK_ROI[3]), 
                          (255, 255, 255), 1)
            
            # 각 객체에 대해 박스, 마스크 생성
            for box, mask, class_id, prob in zip(boxes, masks, cls, probs):
                label = self.model.names[int(class_id)]

                # 'hand' 객체를 'human' 객체로 변경
                if label == 'hand':
                    label = 'human'

                # 클래스에 해당하는 색상 가져오기
                color = self.colors.get(label, (255, 255, 255))  
                
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
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_with_masks, (x1, y1), (x2, y2), color, 2)                     
                cv2.putText(image_with_masks, f'{label} {prob:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # A_ZONE, B_ZONE, C_ZONE ROI 내 일정 시간 이상 'capsule' 객체 인식 확인
                if label == 'capsule':
                    self.robot.A_ZONE, self.robot.A_ZONE_start_time = self.capsule_detect_check(x1, y1, x2, y2, CAPSULE_CHECK_ROI[0], 'A_ZONE', self.robot.A_ZONE, self.robot.A_ZONE_start_time)
                    self.robot.B_ZONE, self.robot.B_ZONE_start_time = self.capsule_detect_check(x1, y1, x2, y2, CAPSULE_CHECK_ROI[1], 'B_ZONE', self.robot.B_ZONE, self.robot.B_ZONE_start_time)
                    self.robot.C_ZONE, self.robot.C_ZONE_start_time = self.capsule_detect_check(x1, y1, x2, y2, CAPSULE_CHECK_ROI[2], 'C_ZONE', self.robot.C_ZONE, self.robot.C_ZONE_start_time)

                # 씰 확인 ROI 내 'capsule_not_label' 객체 인식 확인
                if label == 'capsule_not_label':
                    self.robot.NOT_SEAL = self.seal_remove_check(x1, y1, x2, y2, SEAL_CHECK_ROI, self.robot.NOT_SEAL)

                # Storagy 위의 'cup' 객체를 인식하고 좌표를 저장하는 리스트 생성
                if label == 'cup':
                    self.cup_trash_list, self.cup_trash_list_pixel = self.make_object_list(x1, y1, x2, y2, image_with_masks, self.cup_trash_list, self.cup_trash_list_pixel)

                # Storagy 위의 'cup_holder' 객체를 인식하고 좌표를 저장하는 리스트 생성
                if label == 'cup_holder':
                    self.cup_holder_list, self.cup_holder_list_pixel = self.make_object_list(x1, y1, x2, y2, image_with_masks, self.cup_holder_list, self.cup_holder_list_pixel)

            # Storagy 위에 'cup' 객체가 있을 때 쓰레기 좌표를 저장하고 우선순위 지정
            if self.cup_trash_list:
                self.robot.cup_trash_detected, self.robot.cup_trash_detect_start_time, self.last_cup_trash_center = self.object_detect_order(image_with_masks, self.robot.cup_trash_detected, self.robot.cup_trash_detect_start_time, self.robot.set_cup_trash_coordinates, self.last_cup_trash_center,
                                                                                                                                            self.cup_trash_x, self.cup_trash_y, self.cup_trash_max_y, self.cup_trash_list,
                                                                                                                                            self.cup_trash_x_pixel, self.cup_trash_y_pixel, self.cup_trash_max_y_pixel, self.cup_trash_list_pixel)
            # Storagy 위에 'cup_holder' 객체가 있을 때 컵 홀더 좌표를 저장하고 우선순위 지정
            if self.cup_holder_list:
                self.robot.cup_holder_detected, self.robot.cup_holder_detect_start_time, self.last_cup_holder_center = self.object_detect_order(image_with_masks, self.robot.cup_holder_detected, self.robot.cup_holder_detect_start_time, self.robot.set_cup_holder_coordinates, self.last_cup_holder_center,
                                                                                                                                            self.cup_holder_x, self.cup_holder_y, self.cup_holder_max_y, self.cup_holder_list,
                                                                                                                                            self.cup_holder_x_pixel, self.cup_holder_y_pixel, self.cup_holder_max_y_pixel, self.cup_holder_list_pixel)
            # 로봇 일시정지 기능
            self.pause_robot(image_with_masks, robot_contours, human_contours)

            # 화면 왼쪽 위에 최단 거리 및 로봇 상태 및 ROI 상태 표시
            cv2.putText(image_with_masks, f'Distance: {self.min_distance:.2f}, state: {self.robot.robot_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image_with_masks, f'A_ZONE: {self.robot.A_ZONE}, B_ZONE: {self.robot.B_ZONE}, C_ZONE: {self.robot.C_ZONE}, NOT_SEAL: {self.robot.NOT_SEAL}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image_with_masks, f'cup_trash_detected: {self.robot.cup_trash_detected}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image_with_masks, f'cup_holder_detected: {self.robot.cup_holder_detected}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
        self.cup_trash_detected, self.cup_holder_detected = False, False
        self.cup_trash_detect_start_time, self.cup_holder_detect_start_time = None, None
        self.pressing = False

    def set_cup_trash_coordinates(self, x_mm, y_mm):
        # 컵 쓰레기 좌표 값을 업데이트
        self.cup_trash_x = x_mm
        self.cup_trash_y = y_mm

    def set_cup_holder_coordinates(self, x_mm, y_mm):
        # 컵 홀더 좌표 값을 업데이트
        self.cup_holder_x = x_mm
        self.cup_holder_y = y_mm

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