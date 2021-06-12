import cv2
import numpy as np
import time
import streamlit as st
import matplotlib.pyplot as plt
import tempfile


def web_cam(mirror=True):

    # Display subheading on top of input image

    start_button = st.checkbox('Run WebCam for detection')

    if start_button:

        #load yolo
        net = cv2.dnn.readNet("weights/yolov3.weights", "weights/yolov3.cfg")

        classes = []
        with open("weights/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        #layer_names = net.getLayerNames()
        output_layers = net.getUnconnectedOutLayersNames()
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        # Loading image

        cap = cv2.VideoCapture(0)

        font = cv2.FONT_HERSHEY_PLAIN
        starting_time = time.time()
        frame_id = 0
        frameST = st.empty()

        FRAME_WINDOW = st.image([])

        while True:
            _, frame = cap.read()
            frame_id += 1
            height, width, _ = frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 1/255, (320, 320), (0, 0, 0), swapRB = True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)

            # Showing information on the screen

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.4:
                        # object detected

                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

            items = []  # Array to store label of detected object(s)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(frame, label+" "+confidence, (x, y+30), font, 3, color, 3)
                    items.append(label)  # Add the output label of bounded object

            elapsed_time = time.time() - starting_time
            fps = frame_id / elapsed_time
            cv2.putText(frame, "FPS: "+ str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)

            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame2)

            #cv2.imshow("Video ", frame)
            #frameST.image(frame, channels="BGR")

            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

