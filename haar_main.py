import cv2
import numpy as np

def face2(faceSource):
    #第一步，创建Haar级联器
    facer = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    eye = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    mouth = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_mouth.xml')
    nose = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_nose.xml')

    #第二步，导入人脸识别的图片并将其灰度化
    #img = cv2.imread('./p3.png')
    img = cv2.imread(faceSource)
    #第三步，进行人脸识别
    #[[x,y,w,h]]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #检测出的人脸上再检测眼睛
    faces = facer.detectMultiScale(gray, 1.1, 3)
    i = 0
    j = 0
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_img = img[y:y+h, x:x+w]
        eyes = eye.detectMultiScale(roi_img, 1.1, 3)
        for (x,y,w,h) in eyes:
            cv2.rectangle(roi_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_eye=roi_img[y:y+h, x:x+w]
            eyename = 'eye' + str(j)
            j = j+1 
            #cv2.imshow(eyename, roi_eye)

        i = i+1
        winname = 'face' + str(i)
        #cv2.imshow(winname, roi_img)


    # mouths = mouth.detectMultiScale(gray, 1.1, 3)
    # for (x,y,w,h) in mouths:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # noses = nose.detectMultiScale(gray, 1.1, 3)
    # for (x,y,w,h) in noses:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

    cv2.imshow('img', img)

    cv2.waitKey(10)
    
    
def face():
    #第一步，创建Haar级联器
    facer = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    eye = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    mouth = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_mouth.xml')
    nose = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_nose.xml')

    #第二步，导入人脸识别的图片并将其灰度化
    img = cv2.imread('./p3.png')

    #第三步，进行人脸识别
    #[[x,y,w,h]]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #检测出的人脸上再检测眼睛
    faces = facer.detectMultiScale(gray, 1.1, 3)
    i = 0
    j = 0
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_img = img[y:y+h, x:x+w]
        eyes = eye.detectMultiScale(roi_img, 1.1, 3)
        for (x,y,w,h) in eyes:
            cv2.rectangle(roi_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_eye=roi_img[y:y+h, x:x+w]
            eyename = 'eye' + str(j)
            j = j+1 
            cv2.imshow(eyename, roi_eye)

        i = i+1
        winname = 'face' + str(i)
        cv2.imshow(winname, roi_img)


    # mouths = mouth.detectMultiScale(gray, 1.1, 3)
    # for (x,y,w,h) in mouths:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # noses = nose.detectMultiScale(gray, 1.1, 3)
    # for (x,y,w,h) in noses:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

    cv2.imshow('img', img)

    cv2.waitKey()

#face2("./p3.png")

def main():
    print("Hello World!")
        # 選擇第一隻攝影機
    cap = cv2.VideoCapture(0)

    # cv2.capture

    while(True):
      # 從攝影機擷取一張影像
      ret, frame = cap.read()

      # 顯示圖片
      #cv2.imshow('frame', frame)
      cv2.imwrite("output.jpg", frame)
      face2("./output.jpg")
      #face2("./p3.png")

      # 若按下 q 鍵則離開迴圈
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 釋放攝影機
    cap.release()

    # 關閉所有 OpenCV 視窗
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    