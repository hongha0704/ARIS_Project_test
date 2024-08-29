# S/N : XYZARIS0V3P2311N02
# Robot IP : 192.168.1.192
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

import cv2
import mediapipe as mp
import random
import numpy as np


import cv2
import mediapipe as mp
import numpy as np


class MediaPipe:

    def __init__(self, robot_main):
        # 미디어파이프 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.robot = robot_main

        # 웹캠에서 비디오 스트림 읽기
        self.cap = cv2.VideoCapture(0)

    def mediapipe(self):
        
        # 기준 거리 설정
        base_distance = None

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # BGR 이미지를 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 얼굴 랜드마크 추적
            result = self.face_mesh.process(rgb_frame)

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    # 얼굴 랜드마크 그리기
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

                    # 코 (1)와 오른쪽 얼굴 끝 (234) 랜드마크 좌표 추출
                    nose = face_landmarks.landmark[1]
                    right_cheek = face_landmarks.landmark[234]

                    # 이미지 크기에 맞게 좌표 스케일링
                    h, w, _ = frame.shape
                    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
                    right_cheek_x, right_cheek_y = int(right_cheek.x * w), int(right_cheek.y * h)

                    # 코와 오른쪽 얼굴 끝 사이의 거리 계산
                    distance = np.linalg.norm(np.array([nose_x, nose_y]) - np.array([right_cheek_x, right_cheek_y]))

                    # 기준 거리 설정
                    if base_distance is None:
                        base_distance = distance

                    # 기준 거리와 현재 거리 비교
                    if distance > base_distance * 1.5:
                        self.robot.direction = 'left'
                    elif distance < base_distance * 0.5:
                        self.robot.direction = 'right'
                    else:
                        self.robot.direction = 'center'

            # 반전된 화면에 방향 표시
            flipped_frame = cv2.flip(frame, 1)
            cv2.putText(flipped_frame, f"Direction: {self.robot.direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 결과 영상 출력
            cv2.imshow('MediaPipe Face Mesh', flipped_frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


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

        self.HOST = '192.168.1.167'
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
            print('loop start')
            time.sleep(0.5)
            try:
                print('waiting')
                self.clientSocket.settimeout(10.0)
                self.recv_msg = self.clientSocket.recv(1024).decode('utf-8')
                # try:
                #    self.recv_msg = self.clientSocket.recv(1024).decode('utf-8')
                # except Exception as e:
                #    self.pprint('MainException: {}'.format(e))
                print('\n' + self.recv_msg)
                if self.recv_msg == '':
                    print('here')
                    # continue
                    # pass
                    # break
                    raise Exception('empty msg')
                self.recv_msg = self.recv_msg.split('/')

                if self.recv_msg[0] == 'app_ping':
                    # print('app_ping received')
                    send_msg = 'robot_ping'
                    now_temp = arm.temperatures
                    now_cur = arm.currents
                    send_msg = [
                        {
                            'type': 'A', 'joint_name': 'Base', 'temperature': now_temp[0],
                            'current': round(now_cur[0], 3) * 100
                        }, {
                            'type': 'B', 'joint_name': 'Shoulder', 'temperature': now_temp[1],
                            'current': round(now_cur[1], 3) * 100
                        }, {
                            'type': 'C', 'joint_name': 'Elbow', 'temperature': now_temp[2],
                            'current': round(now_cur[2], 3) * 100
                        }, {
                            'type': 'D', 'joint_name': 'Wrist1', 'temperature': now_temp[3],
                            'current': round(now_cur[3], 3) * 100
                        }, {
                            'type': 'E', 'joint_name': 'Wrist2', 'temperature': now_temp[4],
                            'current': round(now_cur[4], 3) * 100
                        }, {
                            'type': 'F', 'joint_name': 'Wrist3', 'temperature': now_temp[5],
                            'current': round(now_cur[5], 3) * 100
                        }
                    ]
                    try:
                        time.sleep(0.5)
                        self.clientSocket.send(f'{send_msg}'.encode('utf-8'))
                        print('robot_ping')

                    except Exception as e:
                        self.pprint('MainException: {}'.format(e))
                        print('ping send fail')
                    # send_msg = arm.temperatures
                    if self.state == 'ready':
                        print('STATE : ready for new msg')
                    else:
                        print('STATE : now moving')
                else:
                    self.recv_msg[0] = self.recv_msg[0].replace("app_ping", "")
                    if self.recv_msg[0] in ['breath', 'greet', 'farewell' 'dance_random', 'dance_a', 'dance_b',
                                            'dance_c',
                                            'sleep', 'comeon']:
                        print(f'got message : {self.recv_msg[0]}')
                        if self.state == 'ready':
                            self.state = self.recv_msg[0]
                    elif self.recv_msg[0] == 'robot_script_stop':
                        code = self._arm.set_state(4)
                        if not self._check_code(code, 'set_state'):
                            return
                        sys.exit()
                        self.is_alive = False
                        print('program exit')

                    # 픽업존 아이스크림 뺐는지 여부 확인
                    elif self.recv_msg[0].find('icecream_go') >= 0 or self.recv_msg[0].find(
                            'icecream_stop') >= 0 and self.state == 'icecreaming':
                        print(self.recv_msg[0])
                        if self.recv_msg[0].find('icecream_go') >= 0:
                            self.order_msg['makeReq']['latency'] = 'go'
                        else:
                            self.order_msg['makeReq']['latency'] = 'stop'
                            print('000000000000000000000000000000')

                    # 실링 존재 여부 확인

                    if self.recv_msg[0].find('sealing_pass') >= 0 and self.state == 'icecreaming':
                        self.order_msg['makeReq']['sealing'] = 'go'
                        print('socket_go')
                    elif self.recv_msg[0].find('sealing_reject') >= 0 and self.state == 'icecreaming':
                        self.order_msg['makeReq']['sealing'] = 'stop'
                        print('socket_stop')

                    else:
                        # print('else')
                        try:
                            self.order_msg = json.loads(self.recv_msg[0])
                            if self.order_msg['type'] == 'ICECREAM':
                                if self.state == 'ready':
                                    print('STATE : icecreaming')
                                    print(f'Order message : {self.order_msg}')
                                    self.state = 'icecreaming'
                            # else:
                            #    self.clientSocket.send('ERROR : already moving'.encode('utf-8'))
                            else:
                                self.clientSocket.send('ERROR : wrong msg received'.encode('utf-8'))
                        except:
                            pass
                self.recv_msg[0] = 'zzz'

            except Exception as e:
                self.pprint('MainException: {}'.format(e))
                # if e == 'empty msg' :
                #    pass
                # self.connected = False
                print('connection lost')
                while True:
                    time.sleep(2)
                    try:

                        try:
                            self.serverSocket.shutdown(socket.SHUT_RDWR)
                            self.serverSocket.close()
                        except:
                            pass

                        print('socket_making')
                        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

                        self.serverSocket.bind(self.ADDR)
                        print("bind")

                        while True:
                            print('listening')
                            self.serverSocket.listen(1)
                            print(f'reconnecting')
                            try:
                                self.clientSocket, addr_info = self.serverSocket.accept()
                                break

                            except socket.timeout:
                                print('socket.timeout')
                                break

                            except:
                                pass
                        break
                    except Exception as e:
                        self.pprint('MainException: {}'.format(e))
                        print('except')
                        # pass

    # =================================  motion  =======================================
    def motion_breath(self):
        pass

    def motion_sleep(self):  # designed 'sleep'
        try:
            self.clientSocket.send('sleep_start'.encode('utf-8'))
        except:
            print('socket error')

        for i in range(int(1)):
            if not self.is_alive:
                break
            for i in range(int(2)):
                if not self.is_alive:
                    break
                self._angle_speed = 20
                self._angle_acc = 200
                code = self._arm.set_servo_angle(angle=[179.0, -17.7, 29.0, 177.8, 43.8, -1.4], speed=self._angle_speed,
                                                 mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'):
                    return
                self._angle_speed = 5
                self._angle_acc = 5
                code = self._arm.set_servo_angle(angle=[179.0, -10.2, 24.0, 178.2, 39.2, -2.0], speed=self._angle_speed,
                                                 mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'):
                    return
            self._angle_speed = 30
            self._angle_acc = 300
            code = self._arm.set_servo_angle(angle=[179.0, -17.7, 29.0, 177.8, 43.8, -1.4], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            for i in range(int(3)):
                if not self.is_alive:
                    break
                self._angle_speed = 180
                self._angle_acc = 1000
                code = self._arm.set_servo_angle(angle=[179.0, -17.7, 29.0, 199.8, 43.4, -11.0],
                                                 speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'):
                    return
                code = self._arm.set_servo_angle(angle=[179.0, -17.7, 29.0, 157.3, 43.2, 12.7], speed=self._angle_speed,
                                                 mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'):
                    return
            self._angle_speed = 20
            self._angle_acc = 200
            code = self._arm.set_servo_angle(angle=[179.0, -17.7, 29.0, 177.8, 43.8, -1.4], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_pause_time(2)
            if not self._check_code(code, 'set_pause_time'):
                return
        while True:
            try:
                self.clientSocket.send('sleep_finish'.encode('utf-8'))
                break
            except:
                print('socket error')

    def motion_clean_mode(self):
        pass

    def pin_off(self):
        self.clientSocket.send('pin_off_start'.encode('utf-8'))
        # cup_dispenser_up
        code = self._arm.set_cgpio_analog(0, 0)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        code = self._arm.set_cgpio_analog(1, 0)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        # press_up
        code = self._arm.set_cgpio_digital(1, 0, delay_sec=0)
        if not self._check_code(code, 'set_cgpio_digital'):
            return
        self.clientSocket.send('pin_off_finish'.encode('utf-8'))

    def pin_test(self):
        time.sleep(3)
        code = self._arm.set_servo_angle(angle=[179.0, -17.7, 29.0, 177.8, 43.8, -1.4], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_cgpio_digital(0, 1, delay_sec=0)
        if not self._check_code(code, 'set_cgpio_digital'):
            return
        time.sleep(2)
        code = self._arm.set_servo_angle(angle=[179.0, -17.7, 83.3, 177.8, 43.8, -1.4], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        time.sleep(1)
        code = self._arm.set_cgpio_digital(0, 0, delay_sec=0)
        if not self._check_code(code, 'set_cgpio_digital'):
            return
        code = self._arm.set_cgpio_analog(0, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        code = self._arm.set_cgpio_analog(1, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        time.sleep(3)
        code = self._arm.set_cgpio_analog(0, 0)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        time.sleep(3)
        code = self._arm.set_cgpio_analog(1, 0)
        if not self._check_code(code, 'set_cgpio_analog'):
            return

    def joint_state(self):
        while self.is_alive:
            print(f'joint temperature : {arm.temperatures}')
            time.sleep(0.5)
            print(f'joint current : {arm.currents}')
            time.sleep(10)


    def run_chamchamcham(self):

        print("--------------------cham cham cham start--------------------")

        self.direction = 'center'

        self._angle_speed = 100
        self._angle_acc = 200

        code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 180, -62, 90], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return   
        
        time.sleep(1)

        self._angle_speed = 200
        self._angle_acc = 500

        code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 180, -42, 90], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return

        code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 180, -82, 90], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 180, -42, 90], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 180, -82, 90], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 180, -42, 90], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        # 1, 2, 3 중 하나를 무작위로 선택하여 cham_mode 변수에 설정
        robot_direction = random.choice(['left', 'center', 'right'])
        
        # ---------- 로봇 왼쪽, 가운데, 오른쪽 동작  ----------
        if robot_direction == 'left':
            code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 120, -75.5, 115], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return
            human_direction = self.direction

        elif robot_direction == 'center':
            code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 180, -82, 90], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 180, -62, 90], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return
            human_direction = self.direction

        elif robot_direction == 'right':
            code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 240, -75.5, 75], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return
            human_direction = self.direction
            
        print(f"robot_direction: {robot_direction}")
        print(f"human_direction: {human_direction}")

        time.sleep(1)

        if robot_direction == human_direction:   # 졌을 때
            print('You Lose!')
            code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 180, -62, 90], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 90, 0, 90], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 90, -30, 90], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 90, 30, 90], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 90, -30, 90], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 90, 30, 90], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

        elif robot_direction != human_direction: # 이겼을 때
            print('You Win!')

            code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 180, -62, 90], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 180, -100, 90], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            time.sleep(3)
        
        code = self._arm.set_servo_angle(angle=[270, 19.2, 170.6, 180, -62, 90], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return

        print("--------------------cham cham cham end--------------------")



if __name__ == '__main__':
    RobotMain.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    arm = XArmAPI('192.168.1.167', baud_checkset=False)
    robot_main = RobotMain(arm)
    mediapipe_main = MediaPipe(robot_main)

    # 스레드 생성
    robot_thread = threading.Thread(target=robot_main.run_chamchamcham)
    mediapipe_thread = threading.Thread(target=mediapipe_main.mediapipe)

    # 스레드 시작
    robot_thread.start()
    mediapipe_thread.start()

    # 스레드가 끝날 때까지 대기
    robot_thread.join()
    mediapipe_thread.join()