import cv2
import numpy as np
img = np.zeros((500, 500, 3), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "BET", (200, 300), font, 6, (255, 255, 255), 5, cv2.LINE_AA)
cv2.imshow("img", img)
cv2.imwrite("query.jpg", img)
cv2.waitKey(0)

