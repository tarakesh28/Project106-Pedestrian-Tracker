import cv2

# Create our body classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output = cv2.VideoWriter("Output.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))
# Loop once video is successfully loaded
while True:
    # Read first frame
    ret, frame = cap.read()
    #Convert Each Frame into Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,255), 2)   

    cv2.imshow("Pedestrian Detection", frame)
    output.write(frame)
    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

output.write(frame)
cap.release()
cv2.destroyAllWindows()