import time
import cv2
import numpy as np
import streamlit as st
import av


def video_function(my_video):

    # Display subheading on top of input image

    #st.video(my_video)
    start_button = st.button(label="Start", key="start_button")

    if start_button:

        #load yolo
        net = cv2.dnn.readNet("weights/yolov3.weights", "weights/yolov3.cfg")

        classes = []
        with open("weights/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        # Loading image
        cap = cv2.VideoCapture(my_video, apiPreference=0)

        # cap = my_video
        font = cv2.FONT_HERSHEY_PLAIN
        starting_time = time.time()
        frame_id = 0
        frameST = st.empty()
        while True:
            _, frame = cap.read()
            frame_id += 1
            height, width, channels = frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

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
                    if confidence > 0.5:
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

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            # Adjust            confidence         threshold and NMS(Non - Maximum        Suppression) threshold

            # score_threshold = st.sidebar.slider("Confidence threshold", 0.00, 1.00, 0.5, 0.01)
            # nms_threshold = st.sidebar.slider("NMS threshold", 0.00, 1.00, 0.5, 0.01)
            # indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
            # print(indexes)

            items = []  # Array to store label of detected object(s)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 5)
                    cv2.putText(frame, label+" "+str(round(confidence, 2)), (x, y+30), font, 3, color, 5)
                    items.append(label)  # Add the output label of bounded object

            elapsed_time = time.time() - starting_time
            fps = frame_id / elapsed_time
            cv2.putText(frame, "FPS: "+ str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)


            #cv2.imshow("Video ", frame)
            #frameST.image(frame, channels="BGR")

            return av.VideoFrame.from_ndarray(frame, format="bgr24")




        cap.release()
        cv2.destroyAllWindows()