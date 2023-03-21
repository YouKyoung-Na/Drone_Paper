import cv2, glob, os, math

import numpy as np
import matplotlib.pyplot as plt

from rembg import remove
from mpl_toolkits.mplot3d import Axes3D


class Dimension:
    def __init__(self, yoloform, class_number, img, crop):
        self.yoloform = yoloform
        self.class_number = class_number
        self.img = img
        self.crop = crop
    """
    * MAIN FUNCTION, ROTATION!!
    """
    def rotation(self):  # token_folder, save_dir
        txt_yolo = self.yoloform
        
        info = txt_yolo.split()
        txt_yolo = np.array(list(map(float, info)))
        print(txt_yolo)
                                                                        #reader(self.save_dir + '/labels/'+self.token_folder, self.save_dir, 'label')
        img = self.img
                                                                        #cv2.imread(self.save_dir + '/images/' + self.token_folder[:-4] + '.jpg') # 영상 정보만 가져오기

        # remove background
        crop_img = self.crop
                                                                            # cv2.imread(self.save_dir + '/crops/' + self.token_folder[:-4] + '.jpg') # 이미지 불러오기
        seg_img = remove(crop_img) # 후경 삭제 이미지 생성(rembg 라이브러리 사용)
        seg_gray_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY) # 연산량 줄이기 위해 gray scale로 변환

        # 전경 영역의 x, y 정보 추출
        Lx, Ly = [], [] # x, y가 true인 부분을 저장할 list
        for y in range(seg_gray_img.shape[0]):
            for x in range(seg_gray_img.shape[1]):
                if seg_gray_img[y][x] != 0: # 색상 정보가 있을 경우(0이 아닌 경우)
                    Lx.append(x) # Lx list에 x 정보 저장
                    Ly.append(y) # Ly list에 y 정보 저장
        Lx, Ly = np.array(Lx), np.array(Ly) # 영상을 np.array 형태로 변경

        # 공분산 행렬 구하기
        X_cen, Y_cen = Lx - Lx.mean(), Ly - Ly.mean() # 각각 값에 평균값 빼기
        Lxy = np.vstack((X_cen, Y_cen)).T # x행렬과 y행렬 합치기
        dot_Lxy = np.dot(Lxy.T, Lxy) # xy 행렬 dot
        mean_dot_Lxy = dot_Lxy / len(dot_Lxy) # xy 행렬에 dot한 결과에 dot_Lxy의 크기만큼 나누기

        # 고유값 및 고유벡터 구하기
        w, v = np.linalg.eig(mean_dot_Lxy)

        # w가 큰 곳이 주성분 벡터!
        if w[0] > w[1]: eig_vec = v[:][0]
        else: eig_vec = v[:][1]

        # 필요한 정보들 변환
        crop_img_w, crop_img_h = crop_img.shape[1], crop_img.shape[0]
        img_w, img_h = img.shape[1], img.shape[0]

        # 영상에서 처리가 가능한 vector 형태로 변환(좌표 매핑)
        cc_x, cc_y = int(crop_img.shape[1]/2), int(crop_img.shape[0]/2) # crop center
        ce_x, ce_y = int((eig_vec[0] + 1) / 2.0 * crop_img_w), int((1-eig_vec[1]) / 2.0 * crop_img_h) # crop vector 끝점

        ic_x, ic_y = float(txt_yolo[1]) * img_w, float(txt_yolo[2]) * img_h # img center
        ie_x, ie_y = ce_x+(int(ic_x)-cc_x), ce_y+(int(ic_y)-cc_y)

        result = interP(ic_x, ic_y, ie_x, ie_y, img_w, img_h, txt_yolo)
        cc_xy = cc_x, cc_y

        return result, cc_xy





    """
    * MAIN FUNCTION, DEPTH!!
    """
    def depth(self, left_uv, cc_xy, class_number): # token_folder, save_dir, left_uv, cc_xy, class_number
        # 파라미터 정보 및 영상 정보 가져오기
        txt_param = reader('param')
        img = self.img
        print(f'txt_param = {txt_param}')
        # x = K[R|t]X 에서 K, R, t 정보 가져오기
        K = np.array([txt_param[0], txt_param[1], txt_param[2]])
        R = np.array([txt_param[5], txt_param[6], txt_param[7]])
        T = np.array([txt_param[8], txt_param[9], txt_param[10]])

        # R_mat 정보 만들기
        R_mat = cv2.Rodrigues(R)
    #     R_mat_inv = cv2.Rodrigues(-R)

        # Projection Matrix 구하기
        P_mat = np.hstack([R_mat[0], T])
        H = np.dot(K, P_mat)

        K_inv = np.linalg.inv(K) # k의 역행렬 구하기

        # left comma와 right comma 계산하기
        left_c = np.dot(K_inv, [left_uv[0], left_uv[1], 1])
        right_c = np.dot(K_inv, [left_uv[0] + cc_xy[0], left_uv[1] + cc_xy[1], 1])

        # 벡터 만들어서 계산하기
        O = [0, 0, 0]
        u, v = left_c - O, right_c - O # Bounding box....

        # 3차원 벡터를 계산한 후 angle 계산하기
        uv_dot = (u[0]*v[0])+(u[1]*v[1])+(u[2]*v[2])
        uv_sqrt = math.sqrt(u[0]**2 + u[1]**2 + u[2]**2) * math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) 
        angle = np.degrees(np.math.acos(uv_dot/uv_sqrt))

        # depth 연산하기
        drone_size = reader(class_number) # drone class에 따라 정보 가져오기
        seta = 90 - (angle/2) # 삼각함수를 이용해 depth 연산
        D = math.tan(math.pi * (seta/180)) * (drone_size[0][0]/2) # 최종 depth 정보

        return D, left_c, right_c



    """
    * MAIN FUNCTION, VISUALIZATION!!
    """
    def visualization(self, D, left_comma, right_comma): # D, left_comma, right_comma
        ax = plt.figure().add_subplot(projection='3d')

        left =  D * left_comma
        right = D * right_comma

        # camera 위치 출력
        ax.scatter(0, 0, 0, c='k', s=100, label='camera center')

        ax.plot([0,10],[0,0], [0,0], c = 'r')
        ax.plot([0,0], [0,10],[0,0], c = 'b')  
        ax.plot([0,0], [0,0], [0,50],c = 'g')  
		
		
		# object 위치 출력
        ax.scatter(left[0],left[1],left[2], c='pink', s=30)
        ax.scatter(right[0],right[1],right[2], c='pink', s=30)
        ax.plot([left[0],right[0]],[left[1],right[1]],[left[2],right[2]], c = 'pink')
		
		# 차원 크기 세팅
        ax.set_xlim3d(-50, 50)
        ax.set_ylim3d(-50, 50)
        ax.set_zlim3d(-10, 300)


        # save figure
        ax.view_init(-30,-90) # View Angle 조정하기
        plt.savefig('./visual_1.png')

        ax.view_init(-90,-86) # View Angle 조정하기
        plt.savefig('./visual_2.png')

        print("준비...완료")









"""
* SUB FUNCTION(자주 활용되는 경우)
"""
### txt_yolo file reader
def reader(class_number):
	if class_number == 'param':
		folder = './parameters/opencv_camera.txt'
	elif class_number == 0: folder = './drones/0.txt'
	elif class_number == 1: folder = './drones/1.txt'
	elif class_number == 2: folder = './drones/2.txt'
	elif class_number == 3: folder = './drones/3.txt'

	with open(folder, 'r') as f:
		line = f.readlines()
		line = list(map(lambda s: s.strip(), line))
		
	result = []
	for idx in range(len(line)):
		info = line[idx].split()
		try:
			info = np.array(list(map(float, info)))
			result.append(info)
		except: pass
		
	return np.array(result)


# 박스의 끝점 좌표 찾기(4점)
def findComma(txt_yolo, img_w, img_h):
    center_x, center_y, width, height = txt_yolo[1], txt_yolo[2], txt_yolo[3], txt_yolo[4]
    
    # 정규화된 좌표값을 이미지의 크기로 변환
    center_x, center_y, width, height = int(center_x * img_w), int(center_y * img_h), int(width * img_w), int(height * img_h)
    
    # 바운딩 박스의 좌표 계산
    top_left_x, top_left_y = int(center_x - width/2), int(center_y - height/2)
    bottom_right_x, bottom_right_y = int(center_x + width/2), int(center_y + height/2)
    top_right_x, top_right_y = bottom_right_x, top_left_y
    bottom_left_x, bottom_left_y = top_left_x, bottom_right_y
    
    return top_left_x, top_left_y, bottom_right_x, bottom_right_y, top_right_x, top_right_y, bottom_left_x, bottom_left_y





"""
* SUB FUNCTION(특정 함수를 위해 존재하는 경우)
"""
# 선형 방정식의 a와 b 찾기
def liner(x1, y1, x2, y2): # y = ax + b
    try: a = (y2 - y1)/(x2 - x1)
    except: a = 0
    try: b = (x2*y1 - x1*y2)/(x2 - x1)
    except: b = 0
    
    return a, b



# 라인벡터 찾기
def findVecForLine(standard_a, a, txt_yolo, img_w, img_h):
    result, start_end = [], []
    TLx, TLy, BRx, BRy, TRx, TRy, BLx, BLy = findComma(txt_yolo, img_w, img_h)
    if abs(a) > 1: # 기울기 정보를 토대로 접하는 Bounding Box 선 찾기
        aa = TLx - TRx
        bb = TLy - TRy
        result = [aa, bb]
        start_end = [TLx, TLy, TRx, TRy]
    else:
        aa = TRx - BRx
        bb = TRy - BRy
        result = [aa, bb]
        start_end = [TRx, TRy, BRx, BRy]
    return result, start_end


# 교점 찾기
def interP(x1_01, y1_01, x2_01, y2_01, w, h, txt_yolo):
    TLx, TLy, BRx, BRy, TRx, TRy, BLx, BLy = findComma(txt_yolo, w, h) # 각 Bounding Box의 점 찾기
    standard_a, standard_b = liner(txt_yolo[1]*w, txt_yolo[2]*h, TRx, TRy) # 기준 벡터 구하기
    
    # 드론의 rotation vector
    start1, end1 = np.array([x1_01, y1_01]), np.array([x2_01, y2_01])
    vector1 = end1 - start1
    
    # standard vector 찾기
    a, b = liner(x1_01, y1_01, x2_01, y2_01)
    vector2, start_end = findVecForLine(standard_a, a, txt_yolo, w, h)
    
    # 연립방정식을 위한 행렬과 상수 벡터 계산
    A = np.array([[-vector1[0], vector2[0]], [-vector1[1], vector2[1]]])
    B = np.array([(start_end[0], start_end[1]) - start1])
    # 연립방정식 풀이
    solution = np.linalg.solve(A, B.T)
    # 교점 계산
    intersection = start1 + vector1 * solution[0]
    # 교점 return
    return intersection
    

# img = 0 
# crop = 0
# D = Dimension('2 0.1234 0.23 0.345 0.99', 2, img, crop)
# D.rotation()