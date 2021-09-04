
import cv2
import numpy as np


#events=[i for i in dir(cv2) if 'EVENT' in i]
#print(events)

def click_event(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,' , ',y)
      #  font =cv2.FONT_HERSHEY_TRIPLEX
       # text=str(x) + ' , '+str(y)
      #  cv2.putText(img,text,(x,y),font,.5,(255,255,0),2)
        cv2.circle(img,(x,y),3,(0,0,255),-1)
        points.append((x,y))
        if len(points) >=2:
            cv2.line(img,points[-2],points[-1],(255,0,0),5)
        cv2.imshow('image',img)
    #if event ==cv2.EVENT_RBUTTONDOWN:
    #    blue=img[y , x , 0]
    #    green=img[y,x,1]
    #    red=img[y,x,2]
    #    font = cv2.FONT_HERSHEY_TRIPLEX
    #    text2 = str(blue) + ' , ' + str(green)+ ' , '+str(red)
    #    cv2.putText(img, text2, (x, y), font, .5, (0, 255, 0), 2)
    #    cv2.imshow('image', img)

#img=np.zeros((512,512,3),np.uint8)
img=cv2.imread('lena.jpg',1)
points=[]
cv2.imshow('image',img)
cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
