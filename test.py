# 运动检测：背景减除法
import cv2
import numpy as np
import time
import datetime
import sys
import math
from PIL import Image
from imutils.object_detection import non_max_suppression
from matplotlib import pyplot as plt
from collections import deque

obj = deque(maxlen=100)
obj.appendleft(None)
mono = None
frame_number = 0
colour = ((0, 205, 205), (154, 250, 0), (34, 34, 178), (211, 0, 148), (255, 118, 72), (137, 137, 139))  # 定义矩形颜色
similary = 0
pts = deque(maxlen=100)
pts.appendleft(None)
pts.appendleft(None)
speed = deque(maxlen=100)
direction = deque(maxlen=100)

cap = cv2.VideoCapture('Walk3.mpg')  # 参数为0是打开摄像头，文件名是打开视频

fgbg = cv2.createBackgroundSubtractorMOG2()  # 混合高斯背景建模算法


# fourcc = cv2.VideoWriter_fourcc(*'XVID')#设置保存图片格式
# out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p")+'.avi',fourcc, 10.0, (768,576))#分辨率要和原视频对应


def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


def make_regalur_image(img, size=(256, 256)):
    """我们有必要把所有的图片都统一到特别的规格，在这里我选择是的256x256的分辨率。"""
    return img.resize(size).convert('RGB')


def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    return sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)


def calc_similar(li, ri):
    return sum(hist_similar(l.histogram(), r.histogram()) for l, r in zip(split_image(li), split_image(ri))) / 16.0


def calc_similar_by_file(lf, rf):
    lf = Image.fromarray(cv2.cvtColor(lf, cv2.COLOR_BGR2RGB))
    rf = Image.fromarray(cv2.cvtColor(rf, cv2.COLOR_BGR2RGB))
    li, ri = make_regalur_image(lf), make_regalur_image(rf)
    return calc_similar(li, ri)


def calc_similar_by_path(lf, rf):
    li, ri = make_regalur_image(Image.open(lf)), make_regalur_image(Image.open(rf))
    return calc_similar(li, ri)


def split_image(img, part_size=(64, 64)):
    w, h = img.size
    pw, ph = part_size
    assert w % pw == h % ph == 0
    return [img.crop((i, j, i + pw, j + ph)).copy() for i in range(0, w, pw) \
            for j in range(0, h, ph)]


while True:

    obj_number = 0
    ret, frame = cap.read()  # 读取图片
    fgmask = fgbg.apply(frame)
    retVal, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    timer = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 形态学去噪
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)  # 开运算去噪

    _, contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找前景
    fg = cv2.dilate(fgmask, None, iterations=3)
    fg = cv2.erode(fg, None, iterations=3)
    fg = cv2.bitwise_and(frame, frame, mask=fg)

    for cont in contours:

        Area = cv2.contourArea(cont)  # 计算轮廓面积
        if Area < 200:  # 过滤面积小于10的形状
            continue

        obj_number += 1  # 轮廓计数加一

        # print("{}-prospect:{}".format(obj_number,Area),end="  ") #打印出每个前景的面积

        rect = cv2.boundingRect(cont)  # 提取矩形坐标

        # print("x:{} y:{}".format(rect[0],rect[1]))#打印坐标

        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), colour[obj_number % 6],
                      1)  # 原图上绘制矩形
        cv2.rectangle(fgmask, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0xff, 0xff, 0xff),
                      1)  # 黑白前景上绘制矩形

        centerx = rect[0] + rect[2] * 0.5
        centery = rect[1] + rect[3] * 0.5
        center = (int(centerx), int(centery))
        pts.appendleft(center)
        if pts[1] is None:
            pts[1] = pts[0]
        if pts[2] is None:
            pts[2] = pts[1]

        speedx = (abs(centerx - (pts[1])[0])) * int(fps) / 5
        speedy = (abs(centery - (pts[1])[1])) * int(fps) / 5
        speednow = (int(speedx), int(speedy))
        speed.appendleft(speednow)
        # k1=abs(centery-(pts[1])[1])/(abs(centerx-(pts[1])[0]))
        # k2=abs((pts[1])[1]-(pts[2])[1])/(abs((pts[1])[1]-(pts[2])[0]))
        AB = [centerx, centery, (pts[1])[0], (pts[1])[1]]
        CD = [(pts[1])[0], (pts[1])[1], (pts[2])[0], (pts[2])[1]]
        ang1 = angle(AB, CD)
        direction.appendleft(ang1)

        obj_cap = fg[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        cv2.imwrite('obj_cap.png', obj_cap, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        obj_cap2 = cv2.imread('obj_cap.png')
        '''
        if obj[0] is not None:
            for obs in list(obj):#对之前的所有对象
                if obs is not None:
                    cv2.imwrite('obs.png',obs, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
                    obs2 = cv2.imread('obs.png')
                    similary = calc_similar_by_file(obs2, obj_cap2)#进行比对
                    if similary >= 0.85:#如果比对成功
                        obs = obj_cap#进行替换
                        cv2.imshow('number+'+str(obj_number),obs)#显示对象
                        cv2.putText(frame, "similary:"+str(similary), (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1) #显示相似度
                    else:
                        cv2.imshow('number+'+str(obj_number),obj_cap)#显示对象
                        obj.appendleft(obj_cap)
        else:
            obj.appendleft(obj_cap)
            cv2.imshow('number:'+str(obj_number),obj_cap)#显示对象
        '''
        if mono is None:
            mono = obj_cap2
            frame_number += 1
        else:
            similary = calc_similar_by_file(mono, obj_cap2)  # 进行比对
            if similary >= 0.82:  # 如果比对成功
                mono = obj_cap2  # 进行替换
                cv2.imshow('number+' + str(1), obj_cap)  # 显示对象

            else:
                cv2.imshow('number+' + str(1), obj_cap)  # 显示对象
                mono = obj_cap2
                frame_number += 1

        y = 10 if rect[1] < 10 else rect[1]  # 防止编号到图片之外
        cv2.putText(frame, str(obj_number), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)  # 在前景上写上编号
    for i in range(1, len(pts)):
        if pts[i] is None or pts[i - 1] is None:
            continue
        cv2.line(frame, pts[i], pts[i - 1], (0, 0, 255), 2)

    cv2.putText(frame, "similary:" + str(similary), (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)  # 显示相似度
    cv2.putText(frame, "count:", (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)  # 显示总数
    cv2.putText(frame, str(obj_number), (75, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, 'people:' + str(frame_number), (5, 80), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

    # print("----------------------------")
    cv2.imshow('fg', fg)  # 在原图上标注

    cv2.imshow('frame', frame)  # 在原图上标注
    cv2.imshow('frame2', fgmask)  # 以黑白的形式显示前景和背景
    # cv2.imwrite('3.png',obj[0], [int( cv2.IMWRITE_JPEG_QUALITY), 95])
    # time.sleep(0.2)
    # out.write(frame)
    k = cv2.waitKey(30) & 0xff  # 按esc退出
    if k == 27:
        break

out.release()  # 释放文件
cap.release()
cv2.destoryAllWindows()  # 关闭所有窗口
