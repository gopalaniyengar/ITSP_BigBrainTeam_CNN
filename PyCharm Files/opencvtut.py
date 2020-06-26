import cv2
import imutils

image = cv2.imread("./cat.jpg")
(h,w,d)=image.shape
#cv2.imshow("CAT",image)
cv2.waitKey(0)
print(h)
print(w)
print(d)

(B,G,R)=image[1284,1295]
print("R={},G={},B={}".format(R,G,B))
roi = image[500:800,500:800]
cv2.imshow("ROI", roi)
cv2.waitKey(0)

resized = cv2.resize(image, (200, 200))
cv2.imshow("Fixed Resizing", resized)
cv2.waitKey(0)

# fixed resizing and distort aspect ratio so let's resize the width
# to be 300px but compute the new height based on the aspect ratio
r = 300.0 / w
dim = (300, int(h * r))
resized = cv2.resize(image, dim)
cv2.imshow("Aspect Ratio Resize", resized)
cv2.waitKey(0)

# manually computing the aspect ratio can be a pain so let's use the
# imutils library instead
resized = imutils.resize(image, width=300)
cv2.imshow("Imutils Resize", resized)
cv2.waitKey(0)

# rotation can also be easily accomplished via imutils with less code
rotated = imutils.rotate(resized, -45)
cv2.imshow("Imutils Rotation", rotated)
cv2.waitKey(0)

# OpenCV doesn't "care" if our rotated image is clipped after rotation
# so we can instead use another imutils convenience function to help
# us out
rotated = imutils.rotate_bound(resized, 45)
cv2.imshow("Imutils Bound Rotation", rotated)
cv2.waitKey(0)

# apply a Gaussian blur with a 11x11 kernel to the image to smooth it,
# useful when reducing high frequency noise
for i in range(1):
   if i==0:
        blurred = cv2.GaussianBlur(resized, (11, 11), 0)
   else:
        blurred = cv2.GaussianBlur(blurred, (11, 11), 0)
   cv2.imshow("Blurred", blurred)
   cv2.waitKey(0)
"""
output = resized.copy()
cv2.rectangle(output, (32, 60), (79, 160), (0, 0, 255), 2)
cv2.imshow("Rectangle", output)
cv2.waitKey(0)
cv2.circle(output, (150, 150), 20, (255, 0, 0), -1)
cv2.imshow("Circle", output)
cv2.waitKey(0)
cv2.line(output, (60, 20), (400, 200), (0, 255,0), 5)
cv2.imshow("Line", output)
cv2.waitKey(0)
"""
# convert the image to grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

edgedetected=cv2.Canny(gray,30,150)
cv2.imshow("Edge Detection", edgedetected)
cv2.waitKey(0)

edgedetected=cv2.Canny(resized,30,150)
cv2.imshow("Edge Detection on colored image", edgedetected)
cv2.waitKey(0)