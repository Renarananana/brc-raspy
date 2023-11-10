import cv2
cap = cv2.VideoCapture(0)
while True:
  ret, frame = cap.read()
  cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
