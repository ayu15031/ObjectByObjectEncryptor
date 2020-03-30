import cv2
import numpy as np
shares=int(input("Enter the number of shares you have: "))
n=int(input("No. of objects: "))
list_output=[]
for i in range(0,shares-1):
    list_output.append(cv2.imread("o"+str(i+2)+".png"))
o1=cv2.imread("o1.png")
f1=open("input.txt","r")
for line in f1:
    x, y, w, h= line.split(" ")
    x=int(x)
    y=int(y)
    w=int(w)
    h=int(h)
    for l in range(0,shares-1):
        for k in range(0,w):
            for j in range(0,h):
                o1[y+j,x+k][0]=(int(o1[y+j,x+k][0])+int(list_output[l][y+j,x+k][0]))%256
                o1[y+j,x+k][1]=(int(o1[y+j,x+k][1])+int(list_output[l][y+j,x+k][1]))%256
                o1[y+j,x+k][2]=(int(o1[y+j,x+k][2])+int(list_output[l][y+j,x+k][2]))%256
                #o4[y+j,x+k][0]=(int(o1[y+j,x+k][0])-int(o2[y+j,x+k][0]))%256
                #o4[y+j,x+k][1]=(int(o1[y+j,x+k][1])-int(o2[y+j,x+k][1]))%256
                #o4[y+j,x+k][2]=(int(o1[y+j,x+k][2])-int(o2[y+j,x+k][2]))%256

    n=n-1
    if n==0:
        break

cv2.imshow("output",o1)
cv2.imwrite("output.png",o1)
cv2.waitKey(0)