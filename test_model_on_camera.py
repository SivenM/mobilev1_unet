import func_inference
import cv2

# load model
interpreter = func_inference.load_net('model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# load video
cap = cv2.VideoCapture('20200628_130105.mp4')
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# initialise video writer
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy4.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
count = 0
while True:
    ret, frame = cap.read()
    count += 1
    print("Frame ", count)
    out_frame = func_inference.predict_person(frame, interpreter, input_details, output_details)
    out.write(out_frame)
    c = cv2.waitKey(20)
    if c & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()