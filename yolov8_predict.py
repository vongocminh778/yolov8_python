import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import torch
import threading
import time
import signal
import serial

shared_variable = 0
data_lock = threading.Lock()
terminate_flag = False  # Flag to control thread termination

def detection():
    """
    Detect objects in a live webcam stream using YOLOv8.

    This script initializes a YOLOv8 model, captures video from a webcam, and performs real-time object detection.

    Args:
        None

    Returns:
        None
    """
    global shared_variable, terminate_flag

    # Initialize YOLOv8 model
    model = YOLO("./models/model.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Extract class names
    names = model.model.names
    print(names)

    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Set frame dimensions
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height

    while True:
        if terminate_flag:  # Check the flag to terminate the thread
            break

        # Read a frame from the camera
        ret, img = cap.read()
        img = cv2.flip(img, 0)
        if not ret:
            break

        results = model(img, conf=0.8, stream=True, device=device, verbose=False)

        for r in results:
            boxes = r.boxes.xyxy.cpu()
            clss = r.boxes.cls.cpu().tolist()

            with data_lock:
                shared_variable = 1 if 1.0 in clss else 0

            confidences = r.boxes.conf.cpu().tolist()
            annotator = Annotator(img, line_width=2, example=str(names))

            for box, cls, conf in zip(boxes, clss, confidences):
                annotator.box_label(box, str(names[cls]) + f" {conf:.2f}", color=colors(cls, True))

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            # break
            terminate_flag = True  # Thiết lập biến cờ để tắt luồng

    cap.release()
    cv2.destroyAllWindows()

def send_serial():
    global shared_variable, terminate_flag
    Ser = serial.Serial('/dev/ttyACM0', 115200)
    time.sleep(2)
    connected = Ser.is_open  # Check if the serial connection is open
    if connected:
        print("Serial connection is open.")
    else:
        print("Serial connection is not open. Check your serial port.")

    while not terminate_flag:
        with data_lock:
            data = str(shared_variable)
            if(connected):
                Ser.write(data.encode())
                time.sleep(1)
            print("shared_variable: {}" .format(shared_variable))
        time.sleep(1)

def signal_handler(signal, frame):
    global terminate_flag
    print("Press Ctrl+C: Exiting the program...")
    terminate_flag = True

if __name__ == "__main__":
    # Register a signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    t1 = threading.Thread(target=detection, args=())
    t2 = threading.Thread(target=send_serial, args=())

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Done!")