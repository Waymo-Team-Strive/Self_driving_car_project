import cv2
import functions as func
import numpy as np
import os

path = os.getcwd()+'/assets/'
image_name = 'solidWhiteCurve.jpg'
video_name = 'solidWhiteRight.mp4'
image = func.read_img(path+image_name)
video = cv2.VideoCapture(path+video_name)
# func.show_img(image)
# def pre_proces(img)
image_gray = func.grayscale(image)
height = image.shape[0]
width = image.shape[1]
region_of_interest_vertics = [
    (0, height), (width//2, height//1.8), (width, height)]
image_region = func.region_of_interest(
    image_gray, np.array([region_of_interest_vertics], np.int32))
green = [255, 0, 0]
kernel = np.ones((3, 3), np.int32)
while True:
    ret, frame = video. read()
    gray = func.grayscale(frame)
    height = frame.shape[0]
    width = frame.shape[1]
    region_of_interest_vertics = [
        (0, height), (width//2, height//1.8), (width, height)]
    in_reg_frame = func.region_of_interest(
        gray, np.array([region_of_interest_vertics], np.int32))

    thresh_hold_image = cv2.adaptiveThreshold(
        in_reg_frame, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 10)

    # closed = cv2.morphologyEx(thresh_hold_image, cv2.MORPH_OPEN, kernel)

    # dilated = cv2.dilate(closed, kernel, iterations=1)

    # cannyed = cv2.Canny(dilated, 100, 200)

    contours, h = cv2.findContours(
        thresh_hold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_drawn = cv2.drawContours(
        frame, contours, -1, green, 2, cv2.LINE_8)
#     contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
#     contours_drawn = cv2.drawContours(
#         frame, contours_sorted[:8], -1, green, 2, cv2.LINE_8)
#     rows, cols = (69, 69)
#     [vx, vy, x, y] = cv2.fitLine(
#         contours_sorted[0], cv2.DIST_L2, 0, 0.01, 0.01)
#     lefty = int((-x*vy/vx) + y)
#     righty = int(((cols-x)*vy/vx)+y)
#     test = cv2.line(contours_sorted[0], (cols-1,
#                     righty), (0, lefty), (0, 255, 0), 2)
    cv2. imshow('frame', frame)
    cv2. imshow('thresh_hold_image', thresh_hold_image)
    cv2. imshow('test', contours_drawn)

    if cv2.waitKey(1) == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
