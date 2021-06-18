import cv2
import numpy as np

path = '/solidWhiteCurve.jpg'

white_lower = np.array([np.round(0 / 2), np.round(0.9 * 255), np.round(0.00 * 255)])
white_upper = np.array([np.round(390 / 2), np.round(1 * 255), np.round(1 * 255)])


# Yellow-ish areas in image
# H value must be appropriate (see HSL color space), e.g. within [40 ... 60]
# L value can be arbitrary (we want everything between bright and dark yellow), e.g. within [0.0 ... 1.0]
# S value must be above some threshold (we want at least some saturation), e.g. within [0.35 ... 1.0]
yellow_lower = np.array([np.round(40 / 2), np.round(0.00 * 255), np.round(0.35 * 255)])
yellow_upper = np.array([np.round(60 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])


GREEN = (0, 255, 0)

kernel = np.ones((3, 3), np.uint8)


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def get_image(path):
    image_dir = 'test_images'
    image = cv2.imread(image_dir + path)
    return image


vc = cv2.VideoCapture('test_videos/solidWhiteRight.mp4')
if vc.isOpened():
    response, frame = vc.read()

else:
    response = False

while response:

    response, frame = vc.read()

    if response:
        test_image_copy = frame.copy()
    else:
        break

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hls, white_lower, white_upper)
    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    res = cv2.bitwise_and(hls, test_image_copy, mask=mask)

    blurred = cv2.GaussianBlur(res, (3, 3), 1)

    height = int(test_image_copy.shape[0] * .66)
    width = int(test_image_copy.shape[1] * .5)

    gray_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    inv_gray_image = ~gray_image

    thresh_hold_image = cv2.adaptiveThreshold(inv_gray_image, 230, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 10)

    closed = cv2.morphologyEx(thresh_hold_image, cv2.MORPH_OPEN, kernel)

    dilated = cv2.dilate(closed, kernel, iterations=2)

    cannyed = cv2.Canny(dilated, 180, 200)

    lines = cv2.HoughLinesP(cannyed, rho=3, theta=1 * np.pi / 180, threshold=110, minLineLength=30, maxLineGap=500)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(test_image_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)

    contours, h = cv2.findContours(cannyed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # contours_sorted = sorted(contours, key=lambda cont: cv2.arcLength(cont, True), reverse=True)
    # bottom_top, bbox = sort_contours(contours, method="bottom-to-top")
    # cnt1 = bottom_top[1]
    # cnt2 = bottom_top[0]
    # cnt_top = bottom_top[-1]
    # bbox1 = bbox[0]
    # bbox2 = bbox[1]
    # bbox_top = bbox[-1]
    cv2.imwrite('test.png', cannyed)

    # try:
    #     rect1 = cv2.minAreaRect(cnt1)
    #     rect2 = cv2.minAreaRect(cnt2)
    #     rect_top = cv2.minAreaRect(cnt_top)
    #     box1 = cv2.boxPoints(rect1)
    #     box2 = cv2.boxPoints(rect2)
    #     box_top = cv2.boxPoints(rect_top)
    #     box1 = np.int0(box1)
    #     box2 = np.int0(box2)
    #     box_top = np.int0(box_top)
    #     # rows, cols = test_image_copy.shape[:2]
    #     # [vx1, vy1, x1, y1] = cv2.fitLine(cnt1, cv2.DIST_L2, 0, 0.01, 0.01)
    #     # [vx2, vy2, x2, y2] = cv2.fitLine(cnt2, cv2.DIST_L2, 0, 0.01, 0.01)
    #     # lefty1 = int((-x1*vy1/vx1) + y1)
    #     # righty1 = int(((cols-x1)*vy1/vx1)+y1)
    #     #
    #     # lefty2 = int((-x2*vy2/vx2) + y2)
    #     # righty2 = int(((cols-x2)*vy2/vx2)+y2)
    #     cv2.drawContours(test_image_copy, [box1], 0, (0, 255, 0), 2)
    #     cv2.drawContours(test_image_copy, [box2], 0, (255, 0, 0), 2)
    #     cv2.rectangle(test_image_copy, (box_top[0, 0], box_top[0, 1]), (box1[0, 0], box2[0, 0]), GREEN, -1)
    #     # cv2.repeat(test_image_copy, box_top)
    #     # cv2.line(test_image_copy, (cols-1, righty2), (0, lefty2), (0, 255, 0), 2)
    #
    # except Exception as e:
    #     continue
    # epsilon = 0.1*cv2.arcLength(cnt, True)
    # approx = cv2.approxPolyDP(cnt, epsilon, True)

    cv2.imshow('test', test_image_copy)
    key = cv2.waitKey(1)
    if key == 27:  # exit on ESC
        break

cv2.destroyAllWindows()

