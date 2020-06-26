import cv2
import imutils
import numpy as np

im1 = cv2.imread('handwriting1.jpg')
im1=imutils.resize(im1, width= 600)
im2 = cv2.imread('handwriting2.jpg')
im2=imutils.resize(im2, width= 600)
im3 = cv2.imread('handwriting3.jpg')
im3=imutils.resize(im3, width= 600)
im3 = im3[270:420,50:550]

gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray 1", gray1)
cv2.imshow("Gray 2", gray2)
cv2.imshow("Gray 3", gray3)
cv2.waitKey(0)

thresh1 = cv2.threshold(gray1, 150, 255, cv2.THRESH_BINARY)[1]
thresh2 = cv2.threshold(gray2, 160, 255, cv2.THRESH_BINARY)[1]
thresh3 = cv2.threshold(gray3, 170, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Gray Threshold 1", thresh1)
cv2.imshow("Gray Threshold 2", thresh2)
cv2.imshow("Gray Threshold 3", thresh3)
cv2.waitKey(0)

cnts1, hierarchy1 = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts2, hierarchy2 = cv2.findContours(thresh2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts3, hierarchy3 = cv2.findContours(thresh3.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i=0

for cnt in cnts1:
	x,y,w,h = cv2.boundingRect(cnt)
	#bound the images
	cv2.rectangle(gray1,(x,y),(x+w,y+h),(0,255,0),1)
	if w > 10 and h > 10:
		# save individual images
		cv2.imwrite(str(i) + ".jpg", thresh1[y:y + h, x:x + w])
		i = i + 1
for cnt in cnts2:
	x,y,w,h = cv2.boundingRect(cnt)
	#bound the images
	cv2.rectangle(gray2,(x,y),(x+w,y+h),(0,255,0),1)
for cnt in cnts3:
	x,y,w,h = cv2.boundingRect(cnt)
	#bound the images
	cv2.rectangle(gray3,(x,y),(x+w,y+h),(0,255,0),1)

cv2.imshow("Gray 1 rect", gray1)
cv2.imshow("Gray 2 rect", gray2)
cv2.imshow("Gray 3 rect", gray3)
cv2.waitKey(0)

"""
ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
_,contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	#bound the images
	cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
i=0
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	#following if statement is to ignore the noises and save the images which are of normal size(character)
	#In order to write more general code, than specifying the dimensions as 100,
	# number of characters should be divided by word dimension
	if w>100 and h>100:
		#save individual images
		cv2.imwrite(str(i)+".jpg",thresh1[y:y+h,x:x+w])
		i=i+1
cv2.namedWindow('BindingBox', cv2.WINDOW_NORMAL)
cv2.imshow('BindingBox',im)
"""