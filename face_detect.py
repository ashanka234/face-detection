import cv2
import sys

img_path = sys.argv[1]
cascade_path = "haar.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)

image = cv2.imread(img_path)
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray_img,
    scaleFactor = 1.3,
    minNeighbors = 2,
    minSize = (30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print ("Found {0} faces!".format(len(faces)))

for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)