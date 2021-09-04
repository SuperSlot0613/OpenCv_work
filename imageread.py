import cv2

img=cv2.imread('lena.jpg',-1)

print(img)

cv2.imshow('saurabh',img)

k=cv2.waitKey(0)  #The k variable store the press key which was user pressing

if k==27:  #27 is keyword of esc key
    cv2.destroyAllWindows()
elif k==ord('s'):  # if user press the s key then image save
    cv2.imwrite('lena_name.png',img)
    cv2.destroyAllWindows()

