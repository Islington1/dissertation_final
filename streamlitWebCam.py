import cv2
import av
import numpy as np
from streamlit_webrtc import *


def web_cam1():

    class VideoTransformer(VideoTransformerBase):

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

            net = cv2.dnn.readNet("weights/yolov3.weights", "weights/yolov3.cfg")
            classes = []
            with open("weights/coco.names", "r") as f:
                classes = [line.strip() for line in f.readlines()]


            colors = np.random.uniform(0, 255, size=(len(classes), 3))
            font = cv2.FONT_HERSHEY_PLAIN
            image = frame.to_ndarray(format="bgr24")
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1/255, (320, 320), (0, 0, 0), swapRB = True, crop=False)
            net.setInput(blob)
            output_layers = net.getUnconnectedOutLayersNames()
            outs = net.forward(output_layers)

            (height, width) = image.shape[:2]

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
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
                    cv2.putText(image, label + " " + confidence, (x, y + 30), font, 3, color, 3)
                    items.append(label)  # Add the output label of bounded object

            #cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
            annotated_image, result = (image, items)

            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_streamer(
        key="web-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoTransformer,
        async_processing=True,
    )

