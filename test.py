import cv2
import numpy as np

path = '/solidWhiteCurve.jpg'

GREEN = (0, 255, 0)

#height =
#width =
kernel = np.ones((3, 3), np.uint8)
#kernel_gaussian =


def get_image(path):
    image_dir = 'test_images'
    image = cv2.imread(image_dir + path)
    return image


test_image = get_image(path)
test_image_copy = test_image.copy()

blurred = cv2.GaussianBlur(test_image, (3, 3), 1)

height = int(test_image.shape[0] * .66)

gray_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

inv_gray_image = ~gray_image

cropped = inv_gray_image[height:, :]
cropped_colour = test_image_copy[height:, :]

thresh_hold_image = cv2.adaptiveThreshold(cropped, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 10)

closed = cv2.morphologyEx(thresh_hold_image, cv2.MORPH_OPEN, kernel)

dilated = cv2.dilate(closed, kernel, iterations=1)

cannyed = cv2.Canny(dilated, 100, 200)

contours, h = cv2.findContours(cannyed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#contours_drawn = cv2.drawContours(cropped_colour, contours, -1, GREEN, 2, cv2.LINE_8, h)


contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

#print(len(contours_sorted))

contours_drawn = cv2.drawContours(cropped_colour, contours_sorted[:8], -1, GREEN, 2, cv2.LINE_8)

print(contours_sorted[0][:,0,1].shape)

rows,cols = (69,69)
[vx,vy,x,y] = cv2.fitLine(contours_sorted[0], cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
test = cv2.line(contours_sorted[0],(cols-1,righty),(0,lefty),(0,255,0),2)

# lines = cv2.HoughLines(cannyed, 1, np.pi/180, cannyed.shape[0]-1)
#
# for line in lines:
#     for rho, theta in line:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + 1000*(-b))
#         y1 = int(y0 + 1000*(a))
#         x2 = int(x0 - 1000*(-b))
#         y2 = int(y0 - 1000*(a))
#
#         cv2.line(test_image_copy, (x1, y1+height), (x2, y2+height), GREEN, 2)


cv2.imshow('test', test)
cv2.waitKey()
cv2.destroyAllWindows()


