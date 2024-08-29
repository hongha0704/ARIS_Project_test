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

import threading

from ultralytics import YOLO
import cv2
import numpy as np
from scipy.spatial.distance import cdist

import logging


class YOLOMain:
    def __init__(self, robot_main):
        # 모델 로드
        self.model = YOLO('/home/beakhongha/YOLO_ARIS/train18/weights/best.pt')
 
        # 캘리브레이션 데이터 로드
        calibration_data = np.load('/home/beakhongha/RobotArm/camera_calibration/calibration_data.npz')
        self.mtx = calibration_data['mtx']
        self.dist = calibration_data['dist']

        # 변수 초기화
        self.center_x_mm = None
        self.center_y_mm = None
        self.last_cup_center = None 

        # 카메라 열기
        self.webcam = cv2.VideoCapture(2)  # 웹캠 장치 열기
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 프레임 너비 설정
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 프레임 높이 설정
        self.robot = robot_main

        if not self.webcam.isOpened():  # 웹캠이 열리지 않은 경우
            print("웹캠을 열 수 없습니다. 프로그램을 종료합니다.")  # 오류 메시지 출력
            exit()  # 프로그램 종료

        # 좌표 변환
        self.camera_points =  np.array([
            [482, 211],  # 기준점 1의 카메라 좌표
            [487, 268],  # 기준점 2의 카메라 좌표
            [501, 327],  # 기준점 3의 카메라 좌표
            [425, 211],  # 기준점 4의 카메라 좌표
            [428, 268],  # 기준점 5의 카메라 좌표
            [429, 325],  # 기준점 6의 카메라 좌표
            [363, 212],  # 기준점 7의 카메라 좌표
            [366, 267],  # 기준점 8의 카메라 좌표
            [364, 325],  # 기준점 9의 카메라 좌표
            [113, 206],  # 기준점 10의 카메라 좌표
            [179, 208],  # 기준점 11의 카메라 좌표
            [242, 205],  # 기준점 12의 카메라 좌표
            [236, 328],  # 기준점 13의 카메라 좌표
            [232, 265],  # 기준점 14의 카메라 좌표
            ], dtype=np.float32)
        
        self.robot_points = np.array([
            [-300, -100],   # 기준점 1의 로봇 좌표
            [-300, 0],      # 기준점 2의 로봇 좌표
            [-300, 100],    # 기준점 3의 로봇 좌표
            [-200, -100],   # 기준점 4의 로봇 좌표
            [-200, 0],      # 기준점 5의 로봇 좌표
            [-200, 100],    # 기준점 6의 로봇 좌표
            [-100, -100],   # 기준점 7의 로봇 좌표
            [-100, 0],      # 기준점 8의 로봇 좌표
            [-100, 100],    # 기준점 9의 로봇 좌표
            [300, -100],    # 기준점 10의 로봇 좌표
            [200,-100],     # 기준점 11의 로봇 좌표
            [100, -100],    # 기준점 12의 로봇 좌표
            [100,100],      # 기준점 13의 로봇 좌표
            [100,0],        # 기준점 14의 로봇 좌표
            ], dtype=np.float32)
        
        self.H = self.compute_homography_matrix()


    # 변환 행렬
    def compute_homography_matrix(self):
        H, _ = cv2.findHomography(self.camera_points, self.robot_points)
        print("호모그래피 변환 행렬 H:\n", H)
        return H
    
    def transform_to_robot_coordinates(self, image_points):
        camera_coords = np.array([image_points], dtype=np.float32)
        camera_coords = np.array([camera_coords])
        robot_coords = cv2.perspectiveTransform(camera_coords, self.H)
        # 좌표를 소수점 한 자리로 반올림
        robot_coords = [round(float(coord), 1) for coord in robot_coords[0][0]]
        return robot_coords


    def update_coordinates(self, center_x_mm, center_y_mm):
        # 로봇 인스턴스의 좌표를 설정
        self.robot.set_center_coordinates(center_x_mm, center_y_mm)


    def predict_on_image(self, img, conf):
        result = self.model(img, conf=conf)[0]

        # Detection
        cls = result.boxes.cls.cpu().numpy() if result.boxes else []  # 클래스, (N, 1)
        probs = result.boxes.conf.cpu().numpy() if result.boxes else []  # 신뢰도 점수, (N, 1)
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []   # 박스 좌표, xyxy 형식, (N, 4)

        # Segmentation
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []  # 마스크, (N, H, W)
        
        return boxes, masks, cls, probs  # 예측 결과 반환

    def overlay(self, image, mask, color, alpha=0.5):
        """이미지와 세그멘테이션 마스크를 결합하여 하나의 이미지를 만듭니다."""
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # 마스크를 이미지 크기로 리사이즈
        colored_mask = np.zeros_like(image, dtype=np.uint8)  # 이미지와 같은 크기의 색 마스크 생성
        for c in range(3):  # BGR 각 채널에 대해
            colored_mask[:, :, c] = mask * color[c]  # 마스크를 색상으로 칠함
        
        mask_indices = mask > 0  # 마스크가 적용된 부분의 인덱스
        if mask_indices.any():  # mask_indices가 유효한지 확인
            overlay_image = image.copy()  # 원본 이미지를 복사하여 오버레이 이미지 생성
            overlay_image[mask_indices] = cv2.addWeighted(image[mask_indices], 1 - alpha, colored_mask[mask_indices], alpha, 0)  # 마스크 부분만 밝기 조절
            return overlay_image  # 오버레이된 이미지 반환
        else:
            return image  # 유효하지 않으면 원본 이미지 반환

    def find_contours(self, mask):
        """마스크에서 외곽선을 찾습니다."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    

    # 객체의 현재 위치와 과거 위치의 차이를 비교하기 위한 함수
    def distance_between_points(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


    def segmentation(self):
        
        # 변수 초기화
        self.robot.A_ZONE, self.robot.B_ZONE, self.robot.C_ZONE, self.robot.NOT_SEAL = False, False, False, False   # ROI 내에서 capsule/capsule_not_label 객체가 인식되었는지 여부
        self.robot.A_ZONE_start_time, self.robot.B_ZONE_start_time, self.robot.C_ZONE_start_time = None, None, None # ROI 내에서 capsule 객체가 몇 초 동안 인식되었는지 여부 확인
        self.robot.cup_trash_detected = False    # ROI 내에서 cup(컵 쓰레기) 객체 인식 여부
        self.robot.trash_detect_start_time = None   # ROI 내에서 cup(컵 쓰레기) 객체가 몇 초 동안 인식되었는지 여부 확인
        
        # YOLO 모델의 로깅 레벨 설정
        logging.getLogger('ultralytics').setLevel(logging.ERROR)

        # 라벨별 색상 정의 (BGR 형식)
        colors = {
            'cup': (0, 255, 0),  # 컵: 녹색
            'capsule': (0, 0, 255),  # 캡슐: 빨간색
            'capsule_label': (255, 255, 0),  # 캡슐 라벨: 노란색
            'capsule_not_label': (0, 255, 255),  # 캡슐 비라벨: 청록색
            'robot': (0, 165, 255),  # 로봇: 오렌지색
            'human': (255, 0, 0),  # 인간: 파란색
            'hand': (0, 255, 255)  # 손: 노란색
        }

        # 영구적으로 설정된 ROI 구역
        rois = [(455, 65, 95, 95), (360, 65, 95, 95), (265, 65, 95, 95)]  # A_ZONE, B_ZONE, C_ZONE 순서
        specific_roi = (450, 230, 110, 110)  # Seal check ROI 구역


        # 카메라 작동
        while True:
            ret, frame = self.webcam.read()  # 웹캠에서 프레임 읽기
            if not ret:  # 프레임을 읽지 못한 경우
                print("카메라에서 프레임을 읽을 수 없습니다. 프로그램을 종료합니다.")  # 오류 메시지 출력
                break  # 루프 종료

            # 현재 프레임 예측, 학습모델 신뢰도 설정
            boxes, masks, cls, probs = self.predict_on_image(frame, conf=0.7)

            # 원본 이미지에 마스크 오버레이 및 디텍션 박스 표시
            image_with_masks = np.copy(frame)  # 원본 이미지 복사

            robot_contours = []
            human_contours = []

            # 설정된 ROI를 흰색 바운딩 박스로 그리고 선을 얇게 설정
            for (x, y, w, h) in rois:
                cv2.rectangle(image_with_masks, (x, y), (x + w, y + h), (255, 255, 255), 1)  # 각 ROI를 흰색 사각형으로 그림
            # 특정 ROI를 흰색 바운딩 박스로 그리고 선을 얇게 설정
            cv2.rectangle(image_with_masks, (specific_roi[0], specific_roi[1]), 
                          (specific_roi[0] + specific_roi[2], specific_roi[1] + specific_roi[3]), 
                          (255, 255, 255), 1)  # 특정 ROI를 흰색 사각형으로 그림

            # 각 객체에 대해 박스, 마스크 생성
            for box, mask, class_id, prob in zip(boxes, masks, cls, probs):  # 각 객체에 대해
                label = self.model.names[int(class_id)]  # 클래스 라벨 가져오기

                if label == 'hand':  # 'hand' 객체를 'human' 객체로 변경
                    label = 'human'

                color = colors.get(label, (255, 255, 255))  # 클래스에 해당하는 색상 가져오기
                
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

                # A_ZONE, B_ZONE, C_ZONE ROI 내 capsule 객체 인식 확인
                if label == 'capsule':  # capsule 객체만 확인
                    for i, (rx, ry, rw, rh) in enumerate(rois[:3]):  # 최대 세 개의 ROI만 확인
                        # ROI와 바운딩 박스의 교차 영역 계산
                        intersection_x1 = max(x1, rx)  # 교차 영역의 왼쪽 위 x 좌표
                        intersection_y1 = max(y1, ry)  # 교차 영역의 왼쪽 위 y 좌표
                        intersection_x2 = min(x2, rx + rw)  # 교차 영역의 오른쪽 아래 x 좌표
                        intersection_y2 = min(y2, ry + rh)  # 교차 영역의 오른쪽 아래 y 좌표

                        # 교차 영역의 면적 계산
                        intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

                        # 바운딩 박스의 면적 계산
                        box_area = (x2 - x1) * (y2 - y1)

                        # 교차 영역이 바운딩 박스 면적의 80% 이상인지 여부
                        is_condition_met = intersection_area >= 0.8 * box_area

                        # 교차 영역이 바운딩 박스 면적의 80% 이상일 때만 True로 설정
                        if is_condition_met:
                            current_time = time.time()  # 현재 시간 기록

                            # ROI 내에서 capsule 객체 2초 이상 인식 확인
                            if i == 0:  # 첫 번째 ROI A_ZONE
                                if not self.robot.A_ZONE:
                                    if self.robot.A_ZONE_start_time is None:
                                        self.robot.A_ZONE_start_time = current_time
                                        print('A_ZONE start time set')
                                    elif current_time - self.robot.A_ZONE_start_time >= 2: # 2초 이상 캡슐 인식 시 A_ZONE = True
                                        self.robot.A_ZONE = True
                                    else:
                                        print(f'Waiting for 2 seconds: {current_time - self.robot.A_ZONE_start_time:.2f} seconds elapsed')
                                else:
                                    self.robot.A_ZONE_start_time = current_time  # 상태가 이미 True인 경우, 시작 시간을 현재 시간으로 갱신

                            elif i == 1:  # 두 번째 ROI B_ZONE
                                if not self.robot.B_ZONE:
                                    if self.robot.B_ZONE_start_time is None:
                                        self.robot.B_ZONE_start_time = current_time
                                    elif current_time - self.robot.B_ZONE_start_time >= 2: # 2초 이상 캡슐 인식 시 B_ZONE = True
                                        self.robot.B_ZONE = True
                                    else:
                                        print(f'Waiting for 2 seconds: {current_time - self.robot.B_ZONE_start_time:.2f} seconds elapsed')
                                else:
                                    self.robot.B_ZONE_start_time = current_time  # 상태가 이미 True인 경우, 시작 시간을 현재 시간으로 갱신

                            elif i == 2:  # 세 번째 ROI C_ZONE
                                if not self.robot.C_ZONE:
                                    if self.robot.C_ZONE_start_time is None:
                                        self.robot.C_ZONE_start_time = current_time
                                    elif current_time - self.robot.C_ZONE_start_time >= 2: # 2초 이상 캡슐 인식 시 C_ZONE = True
                                        self.robot.C_ZONE = True
                                    else:
                                        print(f'Waiting for 2 seconds: {current_time - self.robot.C_ZONE_start_time:.2f} seconds elapsed')
                                else:
                                    self.robot.C_ZONE_start_time = current_time  # 상태가 이미 True인 경우, 시작 시간을 현재 시간으로 갱신
                        else:
                            if i == 0:
                                self.robot.A_ZONE_start_time = None
                            elif i == 1:
                                self.robot.B_ZONE_start_time = None
                            elif i == 2:
                                self.robot.C_ZONE_start_time = None

                # 씰 확인 ROI 내 capsule_not_label 객체 인식 확인
                if label == 'capsule_not_label':
                    rx, ry, rw, rh = specific_roi
                    # 특정 ROI와 바운딩 박스의 교차 영역 계산
                    intersection_x1 = max(x1, rx)
                    intersection_y1 = max(y1, ry)
                    intersection_x2 = min(x2, rx + rw)
                    intersection_y2 = min(y2, ry + rh)

                    # 교차 영역의 면적 계산
                    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

                    # 바운딩 박스의 면적 계산
                    box_area = (x2 - x1) * (y2 - y1)

                    # 교차 영역이 바운딩 박스 면적의 80% 이상일 때만 True로 설정
                    if intersection_area >= 0.8 * box_area:
                        self.robot.NOT_SEAL = True

                # Trash mode : ROI 내에 'cup' 객체의 중심좌표가 일정 시간 이상 변동이 없는지 확인
                if label == 'capsule':  # 'capsule' 객체에 대해서만 중심 좌표 계산 및 출력 """추후 cup으로 교체 예정"""
                    # center 좌표(pixel)
                    center_x_pixel = (x2 - x1) / 2 + x1
                    center_y_pixel = (y2 - y1) / 2 + y1

                    # 이미지 좌표로 실세계 좌표 계산
                    image_points = [center_x_pixel, center_y_pixel]
                    world_points = self.transform_to_robot_coordinates(image_points)

                    self.center_x_mm = world_points[0]
                    self.center_y_mm = world_points[1]

                    cv2.putText(image_with_masks, f'Center: ({int(self.center_x_mm)}, {int(self.center_y_mm)})', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 캡슐 중심 좌표 표시
                    self.update_coordinates(self.center_x_mm, self.center_y_mm)

                    # 컵 쓰레기 ROI 영역 내에 있는지 확인
                    roi_x1, roi_y1 = -400.0, -170.0
                    roi_x2, roi_y2 = 400.0, 145.0

                    current_time = time.time()  # 현재 시간 기록
                                        
                    if self.last_cup_center is not None:
                        if (self.center_x_mm is not None and self.center_y_mm is not None and                               # 중심좌표가 존재하면
                            roi_x1 <= self.center_x_mm <= roi_x2 and roi_y1 <= self.center_y_mm <= roi_y2 and               # ROI 영역 내에 중심좌표 위치하면
                            self.distance_between_points((self.center_x_mm, self.center_y_mm), self.last_cup_center) < 10): # 객체의 중심점이 이동하지 않으면

                            if self.robot.trash_detect_start_time is None:
                                self.robot.trash_detect_start_time = current_time
                                print('trash detect start time set')
                            elif current_time - self.robot.trash_detect_start_time >= 2:  # 2초 이상 ROI 내에 존재하고 중심 좌표 변경 없을 시 쓰레기 탐지
                                self.robot.cup_trash_detected = True
                            else:
                                # 탐지 시작하고 0초부터 2초까지 단위로 출력
                                print(f"Capsule detected for {current_time - self.robot.trash_detect_start_time:.2f} seconds")
                        else:
                            self.robot.trash_detect_start_time = None
                    else:
                        self.robot.trash_detect_start_time = None

                    self.last_cup_center = (self.center_x_mm, self.center_y_mm) # 중심좌표가 갱신


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
            if min_distance <= 50 and min_distance_bool and self.robot.pressing == False:
                self.robot.robot_state = 'robot stop'
                self.robot._arm.set_state(3)
            elif min_distance > 50 or not min_distance_bool:
                self.robot.robot_state = 'robot move'
                self.robot._arm.set_state(0)

            # 화면 왼쪽 위에 최단 거리 및 로봇 상태 및 ROI 상태 표시
            cv2.putText(image_with_masks, f'Distance: {min_distance:.2f}, state: {self.robot.robot_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image_with_masks, f'A_ZONE: {self.robot.A_ZONE}, B_ZONE: {self.robot.B_ZONE}, C_ZONE: {self.robot.C_ZONE}, NOT_SEAL: {self.robot.NOT_SEAL}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 마스크가 적용된 프레임 표시
            # cv2.imshow("Webcam with Segmentation Masks and Detection Boxes", image_with_masks)

            # 'q' 키를 누르면 종료
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # 자원 해제
        self.webcam.release()  # 웹캠 장치 해제
        # cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기




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
        self.state = 'stopped'
        self.pressing = False
        self.order_list = []
        self.gritting_list = []

        self.center_x_mm = None
        self.center_y_mm = None

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

    def set_center_coordinates(self, x_mm, y_mm):
        # 좌표 값을 업데이트
        self.center_x_mm = x_mm
        self.center_y_mm = y_mm

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
        self.PORT = 10002
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
        print("accept")


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



    # ============================== motion ==============================

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
        time.sleep(0.5)

        print('motion_place_capsule finish')
        time.sleep(0.5)

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

        print('motion_grab_cup finish')
        time.sleep(0.5)

    def motion_topping(self, order):

        self.toppingAmount = 5

        print('motion_topping start')

        if self.Toping:
            code = self._arm.set_servo_angle(angle=[36.6, -36.7, 21.1, 85.6, 59.4, 44.5], speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
            if order["topping3"] > 0:
                code = self._arm.set_position(*self.position_topping_C, speed=self._tcp_speed,
                                                mvacc=self._tcp_acc, radius=0.0, wait=True)
                if not self._check_code(code, 'set_position'): return

                code = self._arm.set_cgpio_digital(2, 1, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return

                code = self._arm.set_position(z=20, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                                wait=True)
                if not self._check_code(code, 'set_position'): return
                
                code = self._arm.set_pause_time(self.toppingAmount - 3)
                if not self._check_code(code, 'set_pause_time'):
                    return
                
                # self.pressing = True
                # code = self._arm.set_cgpio_digital(3, 1, delay_sec=0)
                # if not self._check_code(code, 'set_cgpio_digital'):
                #     return

                code = self._arm.set_pause_time(2)
                if not self._check_code(code, 'set_pause_time'):
                    return
                
                code = self._arm.set_cgpio_digital(2, 0, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return

                code = self._arm.set_position(z=-20, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc,
                                                relative=True, wait=False)
                if not self._check_code(code, 'set_position'): return

            elif order["topping2"] > 0:
                code = self._arm.set_servo_angle(angle=[55.8, -48.2, 14.8, 86.1, 60.2, 58.7], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=False, radius=20.0)
                if not self._check_code(code, 'set_servo_angle'): return
                
                code = self._arm.set_servo_angle(angle=self.position_topping_B, speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'): return

                code = self._arm.set_cgpio_digital(1, 1, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return
                
                code = self._arm.set_position(z=20, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                                wait=True)
                if not self._check_code(code, 'set_position'): return

                code = self._arm.set_pause_time(self.toppingAmount - 4)
                if not self._check_code(code, 'set_pause_time'):
                    return
                
                # self.pressing = True
                # code = self._arm.set_cgpio_digital(3, 1, delay_sec=0)
                # if not self._check_code(code, 'set_cgpio_digital'):
                #     return

                code = self._arm.set_pause_time(3)
                if not self._check_code(code, 'set_pause_time'):
                    return
                
                code = self._arm.set_cgpio_digital(1, 0, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return

                code = self._arm.set_position(z=-20, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc,
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

                code = self._arm.set_cgpio_digital(0, 1, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return
                
                code = self._arm.set_pause_time(self.toppingAmount - 1)
                if not self._check_code(code, 'set_servo_angle'): return
                
                code = self._arm.set_pause_time(0)
                if not self._check_code(code, 'set_pause_time'):
                    return
                
                # self.pressing = True
                # code = self._arm.set_cgpio_digital(3, 1, delay_sec=0)
                # if not self._check_code(code, 'set_cgpio_digital'):
                #     return
                
                code = self._arm.set_cgpio_digital(0, 0, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return

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

            code = self._arm.set_position(*[232.7, 134.1, 350.1, -22, 82.2, 61.4], speed=self._tcp_speed,
                                            mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
        else:
            code = self._arm.set_servo_angle(angle=self.position_icecream_no_topping, speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return
        self.pressing = True
        code = self._arm.set_cgpio_digital(3, 1, delay_sec=0)
        if not self._check_code(code, 'set_cgpio_digital'):
            return

        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return

        time.sleep(1)
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        
        time.sleep(1)
        
        code = self._arm.set_position(z=15, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return
        print('motion_topping finish')

    def motion_make_icecream(self):

        print('motion_make_icecream start')

        if self.Toping:
            time.sleep(4)
        else:
            time.sleep(7)

        time.sleep(3.5)
        code = self._arm.set_position(z=-20, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return

        time.sleep(3.5)
        code = self._arm.set_position(z=-10, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return
        
        if not self._check_code(code, 'set_pause_time'):
            return

        # code = self._arm.set_position(z=-50, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
        #                               wait=True)
        # if not self._check_code(code, 'set_position'): return
        
        time.sleep(3)
        self.pressing = False
        code = self._arm.set_cgpio_digital(3, 0, delay_sec=0)
        if not self._check_code(code, 'set_cgpio_digital'):
            return

        print('motion_make_icecream finish')
        time.sleep(3)

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
        
        print('motion_trash_capsule finish')
        time.sleep(0.5)

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

    def robot_pause(self):
        if self.robot_state == 'robot stop':
            self._arm.set_state(3)
        else:
            self._arm.set_state(0)


    # ============================= trash mode =============================
    def trash_check_mode(self):

        print('trash_check_mode start')

        self._angle_speed = 50
        self._angle_acc = 50

        self._tcp_speed = 50
        self._tcp_acc = 50

        # ---------- 왼쪽 구역 쓰레기 탐지 ----------
        code = self._arm.set_servo_angle(angle=[180, -95, 25, 186.7, 100, -1.6], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        time.sleep(5)
        if self.cup_trash_detected == True:
            self.cup_trash_detected = False
            self.trash_mode()
        else:
            print(self.cup_trash_detected)
            pass
            
        self._angle_speed = 50
        self._angle_acc = 50

        self._tcp_speed = 50
        self._tcp_acc = 50
        
        # ---------- 오른쪽 구역 쓰레기 탐지 ----------
        code = self._arm.set_servo_angle(angle=[180, 10, 25, 186.7, 75, -1.6], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        time.sleep(5)
        
        if self.cup_trash_detected == True:
            self.cup_trash_detected = False
            self.trash_mode()
        else:
            print(self.cup_trash_detected)
            pass

        print('trash_check_mode finish')


    def trash_mode(self):

        print('trash_mode start')
        
        trash_mode_initial = [180, -27.2, 1.8, 180, 48.1, 180] #angle
        
        self._angle_speed = 100
        self._angle_acc = 100

        self._tcp_speed = 100
        self._tcp_acc = 500

        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(0.5)
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        
        code = self._arm.set_servo_angle(angle=self.position_home, speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return        

        # -------------------- 쓰레기 탐지되면 동작_왼쪽 바깥쪽 --------------------
        if self.center_x_mm <= -300 and self.center_y_mm >= -130 and self.center_y_mm <= 100:
            code = self._arm.set_servo_angle(angle=trash_mode_initial, speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_position(y=self.center_y_mm, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                        wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(z=-100, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                        wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(*[self.center_x_mm+90, self.center_y_mm, 150.6, 180, -77.1, -180], speed=self._tcp_speed,
                                            mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.close_lite6_gripper()
            if not self._check_code(code, 'close_lite6_gripper'):
                return
            
            time.sleep(2)
            
            code = self._arm.set_position(z=100, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                        wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_servo_angle(angle=[180, 14.4, 30, 275.4, 90, 162.7], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[180, 14.4, 30, 275.4, 90, 162.7], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[135, 14.4, 17.3, 270.9, 83.7, 0], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'close_lite6_gripper'):
                return
            
            time.sleep(3)

            code = self._arm.set_servo_angle(angle=[180, 14.4, 30, 275.4, 90, 162.7], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=self.position_home, speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'close_lite6_gripper'):
                return
        else:
            pass

        # -------------------- 쓰레기 탐지되면 동작_왼쪽 안쪽 --------------------
        if self.center_x_mm >= -300 and self.center_x_mm <= -100 and self.center_y_mm >= -130 and self.center_y_mm <= 110:
            code = self._arm.set_servo_angle(angle=trash_mode_initial, speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
            code = self._arm.set_servo_angle(angle=[180, 18.4, 95.2, 180, -70.7, 180], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
            code = self._arm.set_servo_angle(angle=[180, 54.5, 117.5, 180, -77.5, 180], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_position(y=self.center_y_mm, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                        wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(x=80, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                        wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_servo_angle(angle=[0, 0, -30, 0, -15.5, 0], speed=self._angle_speed,
                                            mvacc=self._angle_acc, relative=True, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[0, 7, -1, 0, -2, 0], speed=self._angle_speed,
                                            mvacc=self._angle_acc, relative=True, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            time.sleep(1)

            code = self._arm.close_lite6_gripper()
            if not self._check_code(code, 'close_lite6_gripper'):
                return
            
            time.sleep(3)

            code = self._arm.set_position(z=50, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                        wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_servo_angle(angle=[180, 36.5, 58, 180, -96.9, 180], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[171.3, 19.3, 33.5, 131.9, -91.7, 180], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[134.5, 4.9, 14.1, 92.9, -80.5, 0], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'open_lite6_gripper'):
                return
            
            time.sleep(1)
            
            code = self._arm.set_servo_angle(angle=[178.6, 4.9, 14.1, 87.5, -80.5, 0], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[178.6, -42.5, 14.1, 94.9, -87, -19], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=self.position_home, speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'stop_lite6_gripper'):
                return
        else:
            pass

        # -------------------- 쓰레기 탐지되면 동작_오른쪽 --------------------
        if self.center_x_mm > 100 and self.center_x_mm < 380 and self.center_y_mm >= -130 and self.center_y_mm <= 100:
            code = self._arm.set_servo_angle(angle=[90, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[0, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[0, 0, 28.4, 180, 64.3, 0], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[-47.7, 6.2, 57.3, 16.5, 57.9, 31.1], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_position(*[self.center_x_mm-20, -173, 307.5, -173, 13.3, -87.6], speed=self._tcp_speed,
                                        mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(z=-56.2, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(z=-23, roll=27.8, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(y=40, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(y=-7, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
            if not self._check_code(code, 'set_position'): return
        
            code = self._arm.set_position(z=-41.5, roll=27,radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(y=50, z=-8.5, roll=3.4,radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
            if not self._check_code(code, 'set_position'): return

            if self.center_y_mm >= -10:
                code = self._arm.set_position(y=self.center_y_mm, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
                if not self._check_code(code, 'set_position'): return

            code = self._arm.close_lite6_gripper()
            if not self._check_code(code, 'close_lite6_gripper'):
                return
            
            time.sleep(2)

            code = self._arm.set_position(z=23, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_servo_angle(angle=[22.6, 0, 14.5, 116.1, 75.6, 180], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'open_lite6_gripper'):
                return
            
            time.sleep(1)

            code = self._arm.set_servo_angle(angle=[90, -53.4, 9.5, 157.3, 21.1, 26.6], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=self.position_home, speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'stop_lite6_gripper'):
                return
        else:
            pass

        print('trash_mode finish')


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

                self.trash_check_mode()

                while not (self.A_ZONE or self.B_ZONE or self.C_ZONE):  # 캡슐 인식 대기
                    time.sleep(0.2)
                    print('캡슐 인식 대기중...')
                time.sleep(2)

                self.motion_grab_capsule()
                self.motion_check_sealing()

                count = 0
                while True:
                    # if sealing_check request arrives or 3sec past
                    if self.NOT_SEAL or count >= 3:      # 3초 간 씰 인식
                        print('seal check complete')
                        break
                    time.sleep(0.2)
                    count += 0.2

                if self.NOT_SEAL:
                    self.motion_place_capsule()
                    self.motion_grab_cup()
                    self.motion_topping(order)
                    self.motion_make_icecream()
                    self.motion_serve()
                    self.motion_trash_capsule()
                    self.motion_home()
                    print('icecream finish')

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
                self.cup_trash_detected = False
                self.trash_detect_start_time = None
                time.sleep(1)
            elif self.MODE == 'gritting':
                self.gritting(gender)


if __name__ == '__main__':
    RobotMain.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    arm = XArmAPI('192.168.1.167', baud_checkset=False)
    robot_main = RobotMain(arm)
    yolo_main = YOLOMain(robot_main)

    robot_thread = threading.Thread(target=robot_main.run_robot)
    yolo_thread = threading.Thread(target=yolo_main.segmentation)
    socket_thread = threading.Thread(target=robot_main.socket_connect)

    robot_thread.start()
    yolo_thread.start()
    socket_thread.start()