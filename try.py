import cv2

frame = cv2.imread("/images/2018-12-19 09_39_06/1545212417572821689_R.png")
cv2.imshow('image', frame)
cv2.waitKey(0)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
