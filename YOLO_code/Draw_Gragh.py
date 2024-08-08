# import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc

# # 한글 폰트 설정
# font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
# font_name = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font_name)

# # 데이터 정의
# methods = ['방법1', '방법2(7개 점 매칭)', '방법3(14개 점 매칭)']
# mean_errors = [10.5, 11.4, 8.8]
# error_variances = [67.8, 13.9, 15.2]

# # 그래프 그리기
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # 오차 평균 선 그래프
# ax1.plot(methods, mean_errors, marker='o', linestyle='-', color='skyblue', label='오차 평균')
# ax1.set_xlabel('방법')
# ax1.set_ylabel('오차 평균', color='skyblue')
# ax1.tick_params(axis='y', labelcolor='skyblue')

# # 오차 분산 선 그래프
# ax2 = ax1.twinx()
# ax2.plot(methods, error_variances, marker='o', linestyle='-', color='lightcoral', label='오차 분산')
# ax2.set_ylabel('오차 분산', color='lightcoral')
# ax2.tick_params(axis='y', labelcolor='lightcoral')

# # 제목 및 범례 추가
# plt.title('방법별 오차 평균 및 오차 분산 비교')
# fig.tight_layout()
# fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))

# plt.show()

###########################################################################################################

# import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc

# # 한글 폰트 설정
# font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
# font_name = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font_name)

# # 데이터 정의
# methods = ['방법1', '방법2(7개 점 매칭)', '방법3(14개 점 매칭)']
# mean_errors = [10.5, 11.4, 8.8]
# error_variances = [67.8, 13.9, 15.2]

# # 그래프 그리기
# fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(10, 12))

# # 오차 평균 선 그래프
# ax1.plot(methods, mean_errors, marker='o', linestyle='-', color='skyblue')
# ax1.set_xlabel('방법')
# ax1.set_ylabel('오차 평균')
# ax1.set_title('방법별 오차 평균')
# ax1.grid(True)

# # 오차 분산 선 그래프
# ax2.plot(methods, error_variances, marker='o', linestyle='-', color='lightcoral')
# ax2.set_xlabel('방법')
# ax2.set_ylabel('오차 분산')
# ax2.set_title('방법별 오차 분산')
# ax2.grid(True)

# # 그래프 간격 조정
# plt.tight_layout()
# plt.show()


############################################################################################################

import matplotlib.pyplot as plt

# 카메라 픽셀 좌표 리스트
camera_pixel_points = [
    [482, 211], [487, 268], [501, 327], [425, 211], [428, 268], [429, 325],
    [363, 212], [366, 267], [364, 325], [113, 206], [179, 208], [242, 205],
    [236, 328], [232, 265]
]

# 로봇 좌표 리스트
robot_points = [
    [-300, -100], [-300, 0], [-300, 100], [-200, -100], [-200, 0], [-200, 100],
    [-100, -100], [-100, 0], [-100, 100], [300, -100], [200, -100], [100, -100],
    [100, 100], [100, 0]
]

# 좌표를 x축과 y축으로 분리하는 함수
def split_coords(coords):
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]
    return x_coords, y_coords

# 좌표 분리
x_camera, y_camera = split_coords(camera_pixel_points)
x_robot, y_robot = split_coords(robot_points)

# 그래프 그리기
plt.figure(figsize=(14, 6))

# 카메라 픽셀 좌표 그래프
ax1 = plt.subplot(1, 2, 1)
ax1.scatter(x_camera, y_camera, color='blue', marker='o', label='Camera Pixel Points', s=100)
for i, (x, y) in enumerate(zip(x_camera, y_camera)):
    ax1.text(x, y - 5, f'({x}, {y})', fontsize=10, ha='center', va='bottom')
ax1.set_title('Camera Pixel Points', fontsize=14)
ax1.set_xlabel('X-axis (px)', fontsize=12)
ax1.set_ylabel('Y-axis (px)', fontsize=12)
ax1.legend(fontsize=12)
ax1.grid(True)
ax1.invert_yaxis()  # Y축 방향 반전

# x축, y축 범위 설정
ax1.set_xlim(min(x_camera) - 30, max(x_camera) + 30)
ax1.set_ylim(max(y_camera) + 30, min(y_camera) - 30)

# 로봇 좌표 그래프
ax2 = plt.subplot(1, 2, 2)
ax2.scatter(x_robot, y_robot, color='red', marker='o', label='Robot Points', s=100)
for i, (x, y) in enumerate(zip(x_robot, y_robot)):
    ax2.text(x, y - 5, f'({x}, {y})', fontsize=10, ha='center', va='bottom')
ax2.set_title('Robot Points', fontsize=14)
ax2.set_xlabel('X-axis (mm)', fontsize=12)
ax2.set_ylabel('Y-axis (mm)', fontsize=12)
ax2.legend(fontsize=12)
ax2.grid(True)

# x축, y축 범위 설정 및 반전 적용
ax2.set_xlim(max(x_robot) + 50, min(x_robot) - 50)
ax2.set_ylim(max(y_robot) + 50, min(y_robot) - 50)

# 그래프 출력
plt.tight_layout()
plt.show()
