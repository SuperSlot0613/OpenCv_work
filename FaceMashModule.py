import cv2
import mediapipe as mp
import time



class FaceMeshDetector():

    def __init__(self,staticMode=False,maxFace=2,minDetectionCon=0.5,minTrackCon=0.5):
        self.static_image_mode = staticMode
        self.max_num_faces = maxFace
        self.min_detection_confidence = minDetectionCon
        self.min_tracking_confidence = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMash = self.mpFaceMesh.FaceMesh(self.static_image_mode,self.max_num_faces,self.min_detection_confidence,self.min_tracking_confidence )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))


    def findFaceMesh(self,img,draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.faceMash.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_FACE_OVAL, self.drawSpec, self.drawSpec)
                face=[]
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.putText(img,str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)
                    face.append([x,y])
                faces.append(face)

        return img,faces




def main():
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    pTime = 0
    detector=FaceMeshDetector()

    while True:
        success, img = cap.read()

        img,faces=detector.findFaceMesh(img)

        if len(faces) !=0 :
            print(len(faces));


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


if __name__=="__main__":
    main()