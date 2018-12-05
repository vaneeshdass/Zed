import cv2

# Load an color image in grayscale
img = cv2.imread('/root/pycharm/zed-python/pyzed/zed/2018-12-03 11:14:24/1543835665514763946_L_D.png', 0)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('done')
