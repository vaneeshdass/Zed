import cv2

# Load an color image in grayscale
img = cv2.imread('/root/pycharm/zed-python/pyzed/test.png', 0)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('done')
