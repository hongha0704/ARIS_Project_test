# S/N : XYZARIS0V3P2311N03
# Robot IP : 192.168.1.167
# code_version : 3.1.5.2


#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2022, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
# Notice
#   1. Changes to this file on Studio will not be preserved
#   2. The next conversion will overwrite the file with the same name
#
# xArm-Python-SDK: https://github.com/xArm-Developer/xArm-Python-SDK
#   1. git clone git@github.com:xArm-Developer/xArm-Python-SDK.git
#   2. cd xArm-Python-SDK
#   3. python setup.py install
"""
import sys
import math
import time
import queue
import datetime
import random
import traceback
import threading
from xarm import version
from xarm.wrapper import XArmAPI

from threading import Thread, Event
import socket
import json
import os

from ultralytics import YOLO
import cv2
import numpy as np
import time
from scipy.spatial.distance import cdist
import logging


'''상수 Define'''
ESC_KEY = ord('q')           # 캠 종료 버튼
WEBCAM_INDEX = 2             # 사용하고자 하는 웹캠 장치의 인덱스
FRAME_WIDTH = 640            # 웹캠 프레임 너비
FRAME_HEIGHT = 480           # 웹캠 프레임 높이
CONFIDENCE_THRESHOLD = 0.87  # YOLO 모델의 신뢰도 임계값
DEFAULT_MODEL_PATH = '/home/beakhongha/YOLO_ARIS/train23/weights/best.pt'   # YOLO 모델의 경로

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
            self.robot._arm.set_state(3)
        elif self.min_distance > ROBOT_STOP_DISTANCE or not min_distance_bool:
            self.robot.robot_state = 'robot move'
            self.robot._arm.set_state(0)


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



class RobotMain(object):
    """Robot Main Class"""

    def __init__(self, robot, **kwargs):
        self.alive = True
        self._arm = robot
        self._tcp_speed = 100
        self._tcp_acc = 2000
        self._angle_speed = 20
        self._angle_acc = 500
        self._vars = {}
        self._funcs = {}
        self._robot_init()

        # 로봇 상태, 아이스크림 프레스 작동 여부
        self.state = 'stopped'
        self.pressing = False

        # 주문 정보, 성별 정보 리스트
        self.order_list = []
        self.gritting_list = []

        self.position_home = [179.2, -42.1, 7.4, 186.7, 41.5, -1.6] #angle
        self.position_jig_A_grab = [-257.3, -138.3, 198, 68.3, 86.1, -47.0] #linear
        self.position_jig_B_grab = [-152.3, -129.0, 198, 4.8, 89.0, -90.7] #linear
        self.position_jig_C_grab = [-76.6, -144.6, 198, 5.7, 88.9, -50.1] #linear
        self.position_sealing_check = [-136.8, 71.5, 307.6, 69.6, -73.9, -59] #Linear
        self.position_capsule_place = [234.9, 135.9, 465.9, 133.6, 87.2, -142.1] #Linear
        self.position_before_capsule_place = self.position_capsule_place.copy()
        self.position_before_capsule_place[2] += 25
        self.position_cup_grab = [214.0, -100.2, 145.0, -25.6, -88.5, 95.8] #linear
        self.position_topping_A = [-200.3, 162.8, 359.9, -31.7, 87.8, 96.1] #Linear
        self.position_topping_B = [106.5, -39.7, 15.0, 158.7, 40.4, 16.9] #Angle
        self.position_topping_C = [43.6, 137.9, 350.1, -92.8, 87.5, 5.3] #Linear
        self.position_icecream_with_topping = [168.7, 175.6, 359.5, 43.9, 88.3, 83.3] #Linear
        self.position_icecream_no_topping = [48.4, -13.8, 36.3, 193.6, 42.0, -9.2] #angle
        self.position_jig_A_serve = [-258.7, -136.4, 208.2, 43.4, 88.7, -72.2] #Linear
        self.position_jig_B_serve = [-166.8, -126.5, 200.9, -45.2, 89.2, -133.6] #Linear
        self.position_jig_C_serve = [-63.1, -138.2, 199.5, -45.5, 88.1, -112.1] #Linear
        self.position_capsule_grab = [234.2, 129.8, 464.5, -153.7, 87.3, -68.7] #Linear

    def set_cup_trash_coordinates(self, x_mm, y_mm):
        # 컵 쓰레기 좌표 값을 업데이트
        self.cup_trash_x = x_mm
        self.cup_trash_y = y_mm

    def set_cup_holder_coordinates(self, x_mm, y_mm):
        # 컵 홀더 좌표 값을 업데이트
        self.cup_holder_x = x_mm
        self.cup_holder_y = y_mm

        # Robot init
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'register_count_changed_callback'):
            self._arm.register_count_changed_callback(self._count_changed_callback)

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            self.pprint('err={}, quit'.format(data['error_code']))
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

    # Register state changed callback
    def _state_changed_callback(self, data):
        if data and data['state'] == 4:
            self.alive = False
            self.pprint('state=4, quit')
            self._arm.release_state_changed_callback(self._state_changed_callback)

    # Register count changed callback
    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint('counter val: {}'.format(data['count']))

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint('{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}'.format(label, code,
                                                                                                 self._arm.connected,
                                                                                                 self._arm.state,
                                                                                                 self._arm.error_code,
                                                                                                 ret1, ret2))
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1],
                                       ' '.join(map(str, args))))
        except:
            print(*args, **kwargs)

    @property
    def arm(self):
        return self._arm

    @property
    def VARS(self):
        return self._vars

    @property
    def FUNCS(self):
        return self._funcs

    @property
    def is_alive(self):
        if self.alive and self._arm.connected and self._arm.error_code == 0:
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False

    def position_reverse_sealing_fail(self, linear_jig_position = [-257.3, -138.3, 192.1, 68.3, 86.1, -47.0]):
        reverse_position = linear_jig_position.copy()
        reverse_position[2] = reverse_position[2] - 10
        reverse_position[3] = -reverse_position[3]
        reverse_position[4] = -reverse_position[4]
        reverse_position[5] = reverse_position[5] - 180
        return reverse_position

    def socket_connect(self):

        # self.HOST = '192.168.1.167'
        self.HOST = '127.0.0.1'
        self.PORT = 20002
        self.BUFSIZE = 1024
        self.ADDR = (self.HOST, self.PORT)

        # self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.clientSocket.shutdown(1)
            self.clientSocket.close()
        except:
            pass

        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self
        self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # self.serverSocket.allow_reuse_address = True
        while True:
            try:
                self.serverSocket.bind(self.ADDR)
                print("bind")

                while True:
                    self.serverSocket.listen(1)
                    print(f'[LISTENING] Server is listening on robot_server')
                    time.sleep(1)
                    try:
                        while True:
                            try:
                                self.clientSocket, addr_info = self.serverSocket.accept()
                                print("socket accepted")
                                break
                            except:
                                time.sleep(1)
                                print('except')
                                # break

                        break

                    except socket.timeout:
                        print("socket timeout")

                    except:
                        pass
                break
            except:
                pass
        # self.clientSocket.settimeout(10.0)
        print("accept")
        print("--client info--")
        # print(self.clientSocket)

        self.connected = True
        self.state = 'ready'

        # ------------------- receive msg start -----------
        while self.connected:
            try:
                self.recv_msg = json.loads(self.clientSocket.recv(1024).decode())
                print(self.recv_msg)
                if self.recv_msg["topping1"] != 0 or self.recv_msg["topping2"] != 0 or self.recv_msg["topping3"] != 0:
                    self.order_list.append({"topping1" : self.recv_msg["topping1"], 
                                            "topping2" : self.recv_msg["topping2"], 
                                            "topping3" : self.recv_msg["topping3"]})
                if self.recv_msg["gender"] != "":
                    self.gritting_list.append([self.recv_msg["gender"], int(self.recv_msg["age"])])
            except Exception as e:
                print(e)
                continue


    # =================================  motion  =======================================

    def motion_home(self):

        print('motion_home start')

        code = self._arm.set_cgpio_analog(0, 0)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        code = self._arm.set_cgpio_analog(1, 0)
        if not self._check_code(code, 'set_cgpio_analog'):
            return

        # press_up
        code = self._arm.set_cgpio_digital(3, 0, delay_sec=0)
        if not self._check_code(code, 'set_cgpio_digital'):
            return

        # Joint Motion
        self._angle_speed = 80
        self._angle_acc = 200

        code = self._arm.set_servo_angle(angle=self.position_home, speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return

        print('motion_home finish')

    def motion_grab_capsule(self):

        print('motion_grab_capsule start')

        code = self._arm.set_cgpio_analog(0, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        code = self._arm.set_cgpio_analog(1, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        
        # Joint Motion
        self._angle_speed = 100
        self._angle_acc = 100

        self._tcp_speed = 100
        self._tcp_acc = 1000

        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return
        time.sleep(0.5)

        if self.A_ZONE:
            pass
        else:
            code = self._arm.set_servo_angle(angle=[176, 31.7, 31, 76.7, 91.2, -1.9], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(1)

        if self.A_ZONE:
            code = self._arm.set_servo_angle(angle=[179.5, 33.5, 32.7, 113.0, 93.1, -2.3], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=20.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
            code = self._arm.set_position(*self.position_jig_A_grab, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_servo_angle'): return

        elif self.B_ZONE:
            code = self._arm.set_position(*self.position_jig_B_grab, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return

        elif self.C_ZONE:
            code = self._arm.set_servo_angle(angle=[182.6, 27.8, 27.7, 55.7, 90.4, -6.4], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=20.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_position(*self.position_jig_C_grab, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return

        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(1)

        if self.C_ZONE:
            code = self._arm.set_position(z=150, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                          wait=False)
            if not self._check_code(code, 'set_position'): return
            
            self._tcp_speed = 200
            self._tcp_acc = 1000

            code = self._arm.set_tool_position(*[0.0, 0.0, -90.0, 0.0, 0.0, 0.0], speed=self._tcp_speed,
                                               mvacc=self._tcp_acc, wait=False)
            if not self._check_code(code, 'set_servo_angle'): return
            
        else:
            code = self._arm.set_position(z=100, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                          wait=False)
            if not self._check_code(code, 'set_position'): return
            
        self._angle_speed = 180
        self._angle_acc = 500
            
        code = self._arm.set_servo_angle(angle=[145, -18.6, 10.5, 97.5, 81.4, 145], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        print('motion_grab_capsule finish')

    def motion_check_sealing(self):

        print('motion_check_sealing start')

        self._angle_speed = 200
        self._angle_acc = 200

        code = self._arm.set_position(*self.position_sealing_check, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'): return
        
        print('motion_check_sealing finish')

    def motion_place_fail_capsule(self):

        print('motion_place_fail_capsule start')

        if self.A_ZONE:
            code = self._arm.set_servo_angle(angle=[177.3, 5.5, 12.9, 133.6, 81.3, 183.5], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=20.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_position(*self.position_reverse_sealing_fail(self.position_jig_A_grab), speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return

        elif self.B_ZONE:
            code = self._arm.set_servo_angle(angle=[159.5, 11.8, 22.2, 75.6, 92.8, 186.6], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=20.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
            code = self._arm.set_position(*self.position_reverse_sealing_fail(self.position_jig_B_grab) , speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
        elif self.C_ZONE:
            code = self._arm.set_servo_angle(angle=[176.9, -2.2, 15.3, 69.3, 87.5, 195.5], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=20.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
            code = self._arm.set_position(*self.position_reverse_sealing_fail(self.position_jig_C_grab) , speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(1)
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return
        time.sleep(0.5)

        code = self._arm.set_position(z=100, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=False)
        if not self._check_code(code, 'set_position'): return
        
        print('motion_place_fail_capsule finish')

    def motion_place_capsule(self):

        print('motion_place_capsule start')
        
        code = self._arm.set_servo_angle(angle=[81.0, -10.8, 6.9, 103.6, 88.6, 9.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=40.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_servo_angle(angle=[10, -20.8, 7.1, 106.7, 79.9, 26.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=50.0)
        if not self._check_code(code, 'set_servo_angle'): return
                
        code = self._arm.set_servo_angle(angle=[8.4, -42.7, 23.7, 177.4, 31.6, 3.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=40.0)
        if not self._check_code(code, 'set_servo_angle'): return
                
        code = self._arm.set_servo_angle(angle=[8.4, -32.1, 55.1, 96.6, 29.5, 81.9], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_position(*self.position_before_capsule_place, speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'): return
                
        code = self._arm.set_position(*self.position_capsule_place, speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'): return
        
        code = self._arm.set_cgpio_analog(0, 0)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        code = self._arm.set_cgpio_analog(1, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(2)
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return
        time.sleep(1)

        print('motion_place_capsule finish')
        

    def motion_grab_cup(self):

        print('motion_grab_cup start')

        code = self._arm.set_position(*[233.4, 10.3, 471.1, -172.2, 87.3, -84.5], speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=20.0, wait=False)
        if not self._check_code(code, 'set_position'): return
        
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(1)

        code = self._arm.set_servo_angle(angle=[-2.8, -2.5, 45.3, 119.8, -79.2, -18.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=30.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_position(*[195.0, -96.5, 200.8, -168.0, -87.1, -110.5], speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=10.0, wait=False)
        if not self._check_code(code, 'set_position'): return

        code = self._arm.set_position(*self.position_cup_grab, speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'): return
        
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(2)

        code = self._arm.set_position(z=120, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return
        
        code = self._arm.set_servo_angle(angle=[2.9, -31.0, 33.2, 125.4, -30.4, -47.2], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_cgpio_analog(0, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        code = self._arm.set_cgpio_analog(1, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        time.sleep(0.5)

        print('motion_grab_cup finish')
        

    def motion_topping(self, order):

        self.toppingAmount = 5

        print('motion_topping start')

        if self.Toping:
            code = self._arm.set_servo_angle(angle=[36.6, -36.7, 21.1, 85.6, 59.4, 44.5], speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            # ========== 컵 잡는 위치 위로 변경하는 모션 ==========
            code = self._arm.set_servo_angle(angle=[47.7, -44.2, 10.6, 107.1, 72.6, 50.6], speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'open_lite6_gripper'):
                return
            time.sleep(1.5)
            code = self._arm.close_lite6_gripper()
            if not self._check_code(code, 'close_lite6_gripper'):
                return
            time.sleep(1)
            # ===============================================

            if order["topping3"] > 0:
                code = self._arm.set_position(*self.position_topping_C, speed=self._tcp_speed,
                                                mvacc=self._tcp_acc, radius=0.0, wait=True)
                if not self._check_code(code, 'set_position'): return

                # 토핑 추출
                # code = self._arm.set_cgpio_digital(2, 1, delay_sec=0)
                # if not self._check_code(code, 'set_cgpio_digital'):
                #     return
                
                code = self._arm.set_position(z=30, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                                wait=True)
                if not self._check_code(code, 'set_position'): return
                
                code = self._arm.set_pause_time(self.toppingAmount - 3)
                if not self._check_code(code, 'set_pause_time'):
                    return
                
                self.pressing = True
                code = self._arm.set_cgpio_digital(3, 1, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return

                code = self._arm.set_pause_time(2)
                if not self._check_code(code, 'set_pause_time'):
                    return
                
                code = self._arm.set_cgpio_digital(2, 0, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return

                code = self._arm.set_position(z=-30, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc,
                                                relative=True, wait=False)
                if not self._check_code(code, 'set_position'): return

            elif order["topping2"] > 0:
                code = self._arm.set_servo_angle(angle=[55.8, -48.2, 14.8, 86.1, 60.2, 58.7], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=False, radius=20.0)
                if not self._check_code(code, 'set_servo_angle'): return
                
                code = self._arm.set_servo_angle(angle=self.position_topping_B, speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'): return

                # 토핑 추출
                # code = self._arm.set_cgpio_digital(1, 1, delay_sec=0)
                # if not self._check_code(code, 'set_cgpio_digital'):
                #     return
                
                code = self._arm.set_position(z=30, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                                wait=True)
                if not self._check_code(code, 'set_position'): return
                
                code = self._arm.set_pause_time(self.toppingAmount - 4)
                if not self._check_code(code, 'set_pause_time'):
                    return
                
                self.pressing = True
                code = self._arm.set_cgpio_digital(3, 1, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return

                code = self._arm.set_pause_time(3)
                if not self._check_code(code, 'set_pause_time'):
                    return
                
                code = self._arm.set_cgpio_digital(1, 0, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return

                code = self._arm.set_position(z=-30, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc,
                                                relative=True, wait=False)
                if not self._check_code(code, 'set_position'): return
                
                code = self._arm.set_servo_angle(angle=[87.5, -48.2, 13.5, 125.1, 44.5, 46.2], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=False, radius=10.0)
                if not self._check_code(code, 'set_servo_angle'): return

                code = self._arm.set_position(*[43.6, 137.9, 350.1, -92.8, 87.5, 5.3], speed=self._tcp_speed,
                                                mvacc=self._tcp_acc, radius=10.0, wait=False)
                if not self._check_code(code, 'set_position'): return

            elif order["topping1"] > 0:
                code = self._arm.set_position(*self.position_topping_A, speed=self._tcp_speed,
                                                mvacc=self._tcp_acc, radius=0.0, wait=True)
                if not self._check_code(code, 'set_position'): return

                # 토핑 추출
                # code = self._arm.set_cgpio_digital(0, 1, delay_sec=0)
                # if not self._check_code(code, 'set_cgpio_digital'):
                #     return

                code = self._arm.set_position(z=10, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                                wait=True)
                if not self._check_code(code, 'set_position'): return
                
                code = self._arm.set_pause_time(self.toppingAmount - 1)
                if not self._check_code(code, 'set_servo_angle'): return

                code = self._arm.set_pause_time(0)
                if not self._check_code(code, 'set_pause_time'):
                    return
                
                self.pressing = True
                code = self._arm.set_cgpio_digital(3, 1, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return
                
                code = self._arm.set_cgpio_digital(0, 0, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return
                
                code = self._arm.set_position(z=-10, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                                wait=True)
                if not self._check_code(code, 'set_position'): return

                code = self._arm.set_servo_angle(angle=[130.0, -33.1, 12.5, 194.3, 51.0, 0.0], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'): return
                
                code = self._arm.set_position(*[-38.2, 132.2, 333.9, -112.9, 86.3, -6.6], speed=self._tcp_speed,
                                                mvacc=self._tcp_acc, radius=10.0, wait=False)
                if not self._check_code(code, 'set_position'): return
                
                code = self._arm.set_position(*[43.6, 137.9, 350.1, -92.8, 87.5, 5.3], speed=self._tcp_speed,
                                                mvacc=self._tcp_acc, radius=10.0, wait=False)
                if not self._check_code(code, 'set_position'): return
            
            # code = self._arm.set_position(*self.position_icecream_with_topping, speed=self._tcp_speed,
            #                                 mvacc=self._tcp_acc, radius=0.0, wait=True)
            # if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(*[217.2, 138.5, 377.1, 30.2, 84.3, 104,5], speed=self._tcp_speed,
                                                mvacc=self._tcp_acc, wait=True)
            if not self._check_code(code, 'set_position'): return
            
        else:
            self.pressing = True
            code = self._arm.set_cgpio_digital(3, 1, delay_sec=0)
            if not self._check_code(code, 'set_cgpio_digital'):
                return
            code = self._arm.set_servo_angle(angle=self.position_icecream_no_topping, speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return
        time.sleep(0.5)

        print('motion_topping finish')

    def motion_make_icecream(self):

        print('motion_make_icecream start')

        if self.Toping:
            time.sleep(4)
        else:
            time.sleep(7)

        time.sleep(3)
        code = self._arm.set_position(z=-20, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return

        time.sleep(3)
        code = self._arm.set_position(z=-10, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return
        
        if not self._check_code(code, 'set_pause_time'):
            return
        time.sleep(0.5)
        code = self._arm.set_position(z=-30, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return
        
        time.sleep(2)
        self.pressing = False
        code = self._arm.set_cgpio_digital(3, 0, delay_sec=0)
        if not self._check_code(code, 'set_cgpio_digital'):
            return
        time.sleep(0.5)

        print('motion_make_icecream finish')

    def motion_serve(self):

        print('motion_serve start')

        code = self._arm.set_servo_angle(angle=[18.2, -12.7, 8.3, 90.3, 88.1, 23.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=20.0)
        if not self._check_code(code, 'set_servo_angle'): return

        code = self._arm.set_servo_angle(angle=[146.9, -12.7, 8.3, 91.0, 89.3, 22.1], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return

        self._tcp_speed = 100
        self._tcp_acc = 1000

        if self.A_ZONE:
            code = self._arm.set_position(*self.position_jig_A_serve, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.set_position(z=-18, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                          wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'open_lite6_gripper'):
                return
            time.sleep(1)
            code = self._arm.set_position(*[-256.2, -126.6, 210.1, -179.2, 77.2, 66.9], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'stop_lite6_gripper'):
                return
            time.sleep(0.5)
            code = self._arm.set_position(*[-242.8, -96.3, 210.5, -179.2, 77.2, 66.9], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.set_position(*[-189.7, -26.0, 193.3, -28.1, 88.8, -146.0], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
        elif self.B_ZONE:

            code = self._arm.set_position(*self.position_jig_B_serve, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.set_position(z=-13, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                          wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'open_lite6_gripper'):
                return
            time.sleep(1)
            code = self._arm.set_position(*[-165.0, -122.7, 200, -178.7, 80.7, 92.5], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'stop_lite6_gripper'):
                return
            time.sleep(0.5)
            code = self._arm.set_position(*[-165.9, -81.9, 200, -178.7, 80.7, 92.5], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.set_position(*[-168.5, -33.2, 192.8, -92.9, 86.8, -179.3], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
        elif self.C_ZONE:
            code = self._arm.set_servo_angle(angle=[177.6, 0.2, 13.5, 70.0, 94.9, 13.8], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
            code = self._arm.set_position(*self.position_jig_C_serve, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.set_position(z=-12, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                          wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'open_lite6_gripper'):
                return
            time.sleep(1)

            code = self._arm.set_position(*[-75, -132.8, 208, -176.8, 76.1, 123.0], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'stop_lite6_gripper'):
                return
            time.sleep(0.5)

            code = self._arm.set_position(*[-92.0, -107.5, 208, -176.8, 76.1, 123.0], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(*[-98.1, -52.1, 191.4, -68.4, 86.4, -135.0], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
        time.sleep(0.5)
        code = self._arm.set_servo_angle(angle=[169.6, -8.7, 13.8, 85.8, 93.7, 19.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=10.0)
        if not self._check_code(code, 'set_servo_angle'): return

        self._tcp_speed = 100
        self._tcp_acc = 1000

        print('motion_serve finish')

    def motion_serve_storagy(self):

        print('motion_serve_storagy start')

        code = self._arm.set_servo_angle(angle=[18.2, -5.6, 8.6, 99.6,96.2, 14.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=20.0)
        if not self._check_code(code, 'set_servo_angle'): return

        code = self._arm.set_servo_angle(angle=[142.9, -30, 3.2, 133.4, 68.9, 24.1], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=20.0)
        if not self._check_code(code, 'set_servo_angle'): return

        code = self._arm.set_servo_angle(angle=[208.3, -34.5, 12.6, 159.6, 46.9, 12.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return

        self._tcp_speed = 100
        self._tcp_acc = 1000

        # 컵 좌표값 저장
        cup_x_mm = self.cup_holder_x
        cup_y_mm = self.cup_holder_y

        code = self._arm.set_position(*[cup_x_mm, cup_y_mm+100, 350, 0, 90, -90], speed=self._tcp_speed,
                                                mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'): return

        code = self._arm.set_position(z=-80, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                          wait=True)
        if not self._check_code(code, 'set_position'): return

        time.sleep(0.5)

        code = self._arm.set_position(roll=10, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
        if not self._check_code(code, 'set_position'): return

        print('motion_serve_storagy finish')

        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        
        time.sleep(1.5)

        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return

        code = self._arm.set_position(y=130, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                          wait=False)
        if not self._check_code(code, 'set_position'): return

        code = self._arm.set_servo_angle(angle=[235.5, -17.2, 7.3, 180, 63.8, 0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return

        code = self._arm.set_servo_angle(angle=[98.3, -17.1, 6.3, 102.2, 85, 0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return

        print('motion_serve_storagy finish')

    def motion_trash_capsule(self):

        print('motion_trash_capsule start')

        self._angle_speed = 150
        self._angle_acc = 300

        code = self._arm.set_servo_angle(angle=[51.2, -8.7, 13.8, 95.0, 86.0, 17.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=50.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_servo_angle(angle=[-16.2, -19.3, 42.7, 82.0, 89.1, 55.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        
        code = self._arm.set_servo_angle(angle=[-19.9, -19.1, 48.7, 87.2, 98.7, 60.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_position(*[222.8, 0.9, 470.0, -153.7, 87.3, -68.7], speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'): return
        
        code = self._arm.set_position(*self.position_capsule_grab, speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'): return
        
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(1)

        code = self._arm.set_position(z=30, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return
        
        self._tcp_speed = 100
        self._tcp_acc = 1000

        code = self._arm.set_position(*[221.9, -5.5, 500.4, -153.7, 87.3, -68.7], speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'): return
        
        self._angle_speed = 60
        self._angle_acc = 100

        code = self._arm.set_servo_angle(angle=[-10.7, -2.4, 53.5, 50.4, 78.1, 63.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=10.0)
        if not self._check_code(code, 'set_servo_angle'): return

        self._angle_speed = 160
        self._angle_acc = 1000

        code = self._arm.set_servo_angle(angle=[18.0, 11.2, 40.4, 90.4, 58.7, -148.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        # time.sleep(2)

        code = self._arm.set_servo_angle(angle=[25.2, 15.2, 42.7, 83.2, 35.0, -139.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return

        code = self._arm.set_servo_angle(angle=[18.0, 11.2, 40.4, 90.4, 58.7, -148.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_servo_angle(angle=[25.2, 15.2, 42.7, 83.2, 35.0, -139.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return
        self._angle_speed = 120
        self._angle_acc = 1000

        code = self._arm.set_servo_angle(angle=[28.3, -9.0, 12.6, 85.9, 78.5, 20.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=30.0)
        if not self._check_code(code, 'set_servo_angle'): return

        code = self._arm.set_servo_angle(angle=[149.3, -9.4, 10.9, 114.7, 69.1, 26.1], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=50.0)
        if not self._check_code(code, 'set_servo_angle'): return

        code = self._arm.set_servo_angle(angle=[179.2, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        time.sleep(0.5)

        print('motion_trash_capsule finish')
        

    # ============================= trash mode =============================
    def storagy_trash_mode(self):

        print('storagy_trash_mode start')

        self._angle_speed = 150
        self._angle_acc = 200

        self._tcp_speed = 150
        self._tcp_acc = 500

        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(1)
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        
        code = self._arm.set_servo_angle(angle=self.position_home, speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return  
        
        # 컵 쓰레기를 다 버릴 때 까지 무한루프
        while True:
            # 일정시간 동안 컵 탐지
            count = 0
            while True:
                if self.cup_trash_detected or count >= 5:  
                    print('cup detect finish')
                    break
                time.sleep(0.2)
                print("컵 쓰레기 탐지중...")
                count += 0.2

            # 컵 감지 시 쓰레기 버리는 모션 시작
            if self.cup_trash_detected:
                code = self._arm.set_servo_angle(angle=[270, -15.9, 12.1, 180, 49.9, 0], speed=self._angle_speed,
                                                            mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'): return 

                # 컵 좌표값 저장
                cup_x_mm = self.cup_trash_x
                cup_y_mm = self.cup_trash_y

                code = self._arm.set_position(*[cup_x_mm, -189, 264.5, 180, 77.9, 90], speed=self._tcp_speed,
                                                mvacc=self._tcp_acc, radius=0.0, wait=False)
                if not self._check_code(code, 'set_position'): return

                time.sleep(0.5)

                code = self._arm.set_position(*[cup_x_mm, cup_y_mm+130, 264.5, 180, 77.9, 90], speed=self._tcp_speed,
                                                mvacc=self._tcp_acc, radius=0.0, wait=True)
                if not self._check_code(code, 'set_position'): return

                code = self._arm.close_lite6_gripper()
                if not self._check_code(code, 'close_lite6_gripper'):
                    return
                
                time.sleep(2)

                code = self._arm.set_position(y=30, z=90, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                                    wait=True)
                if not self._check_code(code, 'set_position'): return

                code = self._arm.set_servo_angle(angle=[213, -24.5, 35, 180, 32.7, 0], speed=self._angle_speed,
                                                            mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'): return

                code = self._arm.set_servo_angle(angle=[107.6, -27.4, 7.6, 128.3, 65, 28.4], speed=self._angle_speed,
                                                            mvacc=self._angle_acc, wait=False, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'): return

                code = self._arm.set_servo_angle(angle=[18.7, -16.6, 7.6, 100.6, 88.8, 29.2], speed=self._angle_speed,
                                                            mvacc=self._angle_acc, wait=False, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'): return

                code = self._arm.set_servo_angle(angle=[51, -11.8, 20.1, 177.8, 30.9, 180], speed=self._angle_speed,
                                                            mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'): return

                code = self._arm.open_lite6_gripper()
                if not self._check_code(code, 'open_lite6_gripper'):
                    return
                
                time.sleep(1)

                code = self._arm.set_servo_angle(angle=[18.7, -16.6, 7.6, 100.6, 88.8, 29.2], speed=self._angle_speed,
                                                            mvacc=self._angle_acc, wait=False, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'): return

                code = self._arm.set_servo_angle(angle=[107.6, -27.4, 7.6, 128.3, 65, 28.4], speed=self._angle_speed,
                                                            mvacc=self._angle_acc, wait=False, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'): return

                code = self._arm.set_servo_angle(angle=self.position_home, speed=self._angle_speed,
                                                        mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'): return 

                code = self._arm.stop_lite6_gripper()
                if not self._check_code(code, 'stop_lite6_gripper'):
                    return
                
                time.sleep(0.2)

            # self.cup_trash_detected가 False가 되면 무한루프 break
            if not self.cup_trash_detected:
                break

        print('storagy_trash_mode finish')

    # ============================= gritting =============================

    def motion_greet(self):
        try:
            self.clientSocket.send('greet_start'.encode('utf-8'))
        except:
            print('socket error')

        self._angle_speed = 100
        self._angle_acc = 350

        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 181.5, -1.9, -92.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 180.9, -28.3, -92.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 185.4, 30.8, -94.9], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 180.9, -28.3, -92.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 185.4, 30.8, -94.9], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 180.9, -28.3, -92.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 185.4, 30.8, -94.9], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        try:
            self.clientSocket.send('motion_greet finish'.encode('utf-8'))
        except:
            print('socket error')
        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 181.5, -1.9, -92.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        while True:
            try:
                self.clientSocket.send('motion_greet_finish'.encode('utf-8'))
                break
            except:
                print('socket error')

    def gritting(self, gender) -> None: 
        self._angle_speed = 100
        self._angle_acc = 100

        if gender == "Female":
            code = self._arm.set_servo_angle(angle=[207.4, -15.5, 60.6, 181.7, 46.8, -2.5], speed=self._angle_speed, 
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[247.6, -15.5, 8.9, 87, 78.5, -104.7], speed=self._angle_speed, 
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[179.2, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed, 
                                                mvacc=self._angle_acc, wait=True, radius=0.0) # home
            if not self._check_code(code, 'set_servo_angle'):
                return
            
        elif gender == "Male":
            code = self._arm.set_servo_angle(angle=[265.4, -17.3, 105.2, 186.8, -17.1, 0], speed=self._angle_speed, 
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[265.4, -9.6, 37.1, 179.8, -6, -8.2], speed=self._angle_speed, 
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[265.4, -17.3, 105.2, 186.8, -23.1, 0], speed=self._angle_speed, 
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[179.2, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,  
                                                mvacc=self._angle_acc, wait=True, radius=0.0) # home
            if not self._check_code(code, 'set_servo_angle'):
                return
            
        else:
            self.motion_greet()


    # ============================= main =============================
    def run_robot(self):

        self.Toping = True

        while self.is_alive:
            if self.order_list != []:
                self.MODE = 'icecreaming'
                raw_order = self.order_list.pop(0)
                order = raw_order

            elif self.gritting_list != []:
                self.MODE = 'gritting'
                data = self.gritting_list.pop(0)
                gender = data[0]
                age = data[1]
            else:
                self.MODE = 'ready'

            # --------------Joint Motion : icecream start--------------------
            if self.MODE == 'icecreaming':
                print('icecream start')
                time.sleep(4)
                self.motion_home()

                self.storagy_trash_mode()

                # 캡슐 인식 대기
                while not (self.A_ZONE or self.B_ZONE or self.C_ZONE):
                    time.sleep(0.2)
                    print('캡슐 인식 대기중...')
                time.sleep(2)

                self.motion_grab_capsule()
                self.motion_check_sealing()

                # 일정 시간 동안 씰 제거 여부 인식
                count = 0
                while True:
                    if self.NOT_SEAL or count >= 3:      
                        print('seal check complete')
                        break
                    time.sleep(0.2)
                    count += 0.2

                # 씰 제거 확인 시 아이스크림 제조
                if self.NOT_SEAL:
                    self.motion_place_capsule()
                    self.motion_grab_cup()
                    self.motion_topping(order)
                    self.motion_make_icecream()
                    self.motion_serve_storagy()
                    self.motion_trash_capsule()
                    self.motion_home()
                    print('icecream finish')

                # 씰 제거 확인 안될 시 캡슐 return
                else:
                    self.motion_place_fail_capsule()
                    self.motion_home()
                    self.order_list.insert(0, raw_order)
                    print('please take off the seal')

                code = self._arm.stop_lite6_gripper()
                if not self._check_code(code, 'stop_lite6_gripper'):
                    return
                
                # -------------- 동작 종류 후 변수 초기화 --------------
                self.A_ZONE, self.B_ZONE, self.C_ZONE, self.NOT_SEAL = False, False, False, False
                self.A_ZONE_start_time, self.B_ZONE_start_time, self.C_ZONE_start_time = None, None, None
                self.cup_trash_detected, self.cup_holder_detected = False, False
                self.cup_trash_detect_start_time, self.cup_holder_detect_start_time = None, None
                time.sleep(1)
            
            elif self.MODE == 'gritting':
                self.gritting(gender)


if __name__ == '__main__':
    RobotMain.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    arm = XArmAPI('192.168.1.167', baud_checkset=False)
    robot_main = RobotMain(arm)
    yolo_main = YOLOMain(robot_main)

    robot_thread = threading.Thread(target=robot_main.run_robot)
    yolo_thread = threading.Thread(target=yolo_main.run_yolo)
    socket_thread = threading.Thread(target=robot_main.socket_connect)

    robot_thread.start()
    yolo_thread.start()
    socket_thread.start()