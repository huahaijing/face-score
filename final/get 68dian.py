# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:26:58 2019

@author: xiong
"""

import dlib                     #人脸识别的库dlib
import numpy as np              #数据处理的库numpy
import cv2                      #图像处理的库OpenCv
import copy
import pandas as pd


class face:
    
    def __init__(self,i):
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.i = i
        
        self.file = "D:\\My Course\\Data_Collection\\"+str(i)+".jpg"
        #"D:\My Course\统计软件\统计软件分析\final\data2\1.jpg"
        #self.file = '4.jpg'
        self.picture = cv2.imread(self.file)
        
        #self.learn()
        
    def learn(self):
        im_rd = self.picture
        im_rd = copy.deepcopy(self.picture)
        img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
        # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数rects
        faces = self.detector(img_gray, 0)

        # 待会要显示在屏幕上的字体
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if(len(faces)!=0):

                # 对每个人脸都标出68个特征点
                for i in range(len(faces)):
                    # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                    for k, d in enumerate(faces):
                        # 用红色矩形框出人脸
                        cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))
                        # 计算人脸热别框边长
                        self.face_width = d.right() - d.left()

                        # 使用预测器得到68点数据的坐标
                        shape = self.predictor(im_rd, d)
                        self.shape = shape
                        # 圆圈显示每个特征点
                        for i in range(68):
                            cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                            #cv2.putText(im_rd, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            #            (255, 255, 255))
                        for index, pt in enumerate(shape.parts()):
                            img = im_rd
                            #print('Part {}: {}'.format(index, pt))
                            pt_pos = (pt.x, pt.y)
                            cv2.circle(img, pt_pos, 1, (255, 0, 0), 2)
                            #利用cv2.putText输出1-68
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(img, str(index+1),pt_pos,font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

                            
        #cv2.imshow("camera", im_rd)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        self.dian68 = np.zeros((68,3))
        for index, pt in enumerate(self.shape.parts()):
            
            self.dian68[index] = [index,pt.x,pt.y]
        
        list= self.dian68

        column=['i','x','y']
        
        test=pd.DataFrame(columns=column,data=list)
        
        test.to_csv('D:\\My Course\\统计软件\\统计软件分析\\final\\68dian\\'+str(self.i)+".csv")

        
    
    #def 
    
for i in range(1,12):
    a = face(i)
    a.learn()


        