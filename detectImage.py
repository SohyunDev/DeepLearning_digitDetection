import cv2
import predict

img_data = input("choose data number 1~4 : ")
data_name = ""
if img_data == '1':
    data_name = "data/test1.jpeg"
elif img_data == '2':
    data_name = "data/test2.jpeg"
elif img_data == '3':
    data_name = "data/test3.jpeg"
elif img_data == '4':
    data_name = "data/test4.jpeg"
else :
    print("wrong input")

# Loading Image and Preprocessing
img_color = cv2.imread(data_name)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)
ret, im_threshold = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)

# draw contours and split each digits
img_contour, contours, hier = cv2.findContours(im_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(contour) for contour in contours]

# sort rects in order
def rectsort(rect):
    return rect[0]
rects_sorted = sorted(rects, reverse=False, key=rectsort)

# predict each digits
predicted_digit=[]
for rect in rects_sorted:
    #cv2.rectangle(img_color, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]),(0,255,0),3)
    leng = int(rect[3]*1.6)
    pt1 = int(rect[1]+rect[3] //2 - leng //2)
    pt2 = int(rect[0]+rect[2]//2-leng//2)
    roi = im_threshold[pt1:pt1+leng, pt2:pt2+leng]
    roi = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3,3))
    predicted_digit.append(predict.predict(roi))

# get results
result = "".join(map(str, predicted_digit))
print("Predicted digit : " + result)

# show Image
cv2.imshow("img_color", img_color)
cv2.waitKey()
