import cv2
from cvzone.HandTrackingModule import HandDetector
import time

cap=cv2.VideoCapture(0)
cTime=0
pTime = 0

detector=HandDetector(detectionCon=0.8,maxHands=2)

while True:
    success,img=cap.read()

    hands,img=detector.findHands(img)

    print(len(hands))
    #int hands list hand=dist(lmlist,bbox,center,type)

    if hands:
        #hand 1
        hand1=hands[0]
        lmList1=hand1["lmList"]  #This is list of 21 landmarks of hands
        bbox1=hand1["bbox"] #Bounding box info x,y,w,h
        centerPoint1=hand1["center"] #center of the hand cx,cy
        handType1=hand1["type"] #hand type left or righr

        #print(len(lmList1),lmList1)
        #print(handType1)
        fingers1=detector.fingersUp(hand1)
        #length,info,img=detector.findDistance(lmList1[8],lmList1[12],img)

    if len(hands)==2:
        hand2 = hands[1]
        lmList2 = hand2["lmList"]  # This is list of 21 landmarks of hands
        bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
        centerPoint2 = hand2["center"]  # center of the hand cx,cy
        handType2 = hand2["type"]  # hand type left or righr

        #print(handType2)
        fingers2 = detector.fingersUp(hand2)
        #print(fingers1,fingers2)

        #length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
        length, info, img = detector.findDistance(centerPoint1,centerPoint2, img)




    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image",img)

    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break