import cv2
import mediapipe as mp
import time

wCam,hCam=640,480
cap=cv2.VideoCapture("Video/5.mp4")
cap.set(3,wCam)
cap.set(4,hCam)

mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMash=mpFaceMesh.FaceMesh(max_num_faces=3)
drawSpec=mpDraw.DrawingSpec(thickness=2,circle_radius=1,color=(0,255,0))

pTime=0
while True:
    success, img = cap.read()

    if success:
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        results = faceMash.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_FACE_OVAL, drawSpec, drawSpec)

                for id,lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih,iw,ic=img.shape
                    x,y=int(lm.x*iw),int(lm.y*ih)
                    print(id,x,y)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS {int(fps)}', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    if success:
        cv2.imshow("Image", img)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break