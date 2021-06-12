from PIL import Image
import time
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
from web_cam_yolo import *
from yolo_video_detection import *

def main():
    selected_box = st.sidebar.selectbox(
        'Choose one of the following media format',
        ('Welcome', 'Image Object Detection', 'Video', 'Web Cam')
    )

    if selected_box == 'Welcome':
        welcome()

    if selected_box == 'Image Object Detection':
        object_detection()

    if selected_box == 'Video':
        video_detection()

    if selected_box == 'Web Cam':
        web_video_detection()


def welcome():
    st.title('Object Detection Using Deep Learning Algorithm (YOLO)')

    st.write('In this app, we have used YOLOv3, a real-time object detection algorithm that uses a fully'
             ' convolutional network, along with a Darknet backbone, which performs feature extraction')

    st.header('A simple app that shows object detection on image, video and live video. ')

    st.subheader(' You can choose the options from the left.')
    st.write('See the sample of object detection in an Image')
    sample_image = Image.open("images//sample_image.jpg")
    st.image(sample_image)


def video_detection():

    st.header('Detecting Objects in a Video')
    st.write('We will be processing the videos using the pre-trained weights on COCO dataset on 80 classes.')
    st.subheader("Object Detection is done using YOLO V3 Model")
    st.write('The approach is quite similar to detecting images with YOLO.'
             ' We get every frame of a video like an image and detect objects at that frame using yolo.'
             ' Then draw the boxes, labels and iterate through all the frame in a given video.'
             ' Adjust the confidence and nms threshold to see how the algorithms detections change. '
            )

    choice = st.radio("", ("See an illustration", "Upload Video of your choice"))

    if choice == "Upload Video of your choice":
        st.subheader("Input Video")
        video_file = st.file_uploader("Upload", type=['mp4'])

        if video_file is not None:  # if a file has been uploaded
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            video_function(tfile.name)

        # If user selects 1st option
    elif choice == "See an illustration":
       video_function("images/test1.mp4")


def object_detection():

    st.header('Object Detection In an Image')

    st.write('Finding all the objects in an image and drawing the so-called bounding boxes around them, is the main motive. ')

    #st.subheader("Object Detection is done using YOLO Models")
    choice = st.radio("", ("See an illustration", "Choose an image of your choice"))
    # streamlit.radio() inserts a radio button widget

    # If user selects 2nd option:
    if choice == "Choose an image of your choice":
        image_file = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'])
        if image_file is not None:  # if a file has been uploaded
            my_img = Image.open(image_file)  # open the image
            # perform object detection on selected image
            image_function(my_img)
        # If user selects 1st option
    elif choice == "See an illustration":
        # display the example image
        my_img = Image.open("images//test.jpeg")
        # perform object detection on the example image
        image_function(my_img)


def web_video_detection():

    st.header('Object Detection from the Web Cam')
    st.write('Creates an instance of VideoCapture with argument as device index or the name of a video file. Pass 0 as the device index for the camera'
             '. Once the instance of VideoCapture is created, you can capture the video frame-by-frame and detect object.')

    #web_cam();
    web_cam_yolo_fn();



def image_function(my_img):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    column1, column2 = st.beta_columns(2)
    # Display subheading on top of input image
    column1.subheader("Input image")  # streamlit.subheader()

    st.text("")  # streamlit.text() writes preformatted and fixed-width text
    # Display the input image using matplotlib
    plt.figure(figsize=(20, 20))
    plt.imshow(my_img)
    column1.pyplot(use_column_width=True)

    # load yolo
    net = cv2.dnn.readNet("weights/yolov3.weights", "weights/yolov3.cfg")

    # Initialize an array to store output labels
    classes = []
    with open("weights/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # agree = st.sidebar.checkbox(" All Classes")
    # if agree:
    #     st.checkbox("Great", value=True)

    # strip() method removes leading and trailing spaces from the label strings
    layer_names = net.getLayerNames()
    # Store the names of model’s layers obtained using getLayerNames() of OpenCV
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # getUnconnectedOutLayers() returns indexes of layers with unconnected output
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # RGB values selected randomly from 0 to 255 using np.random.uniform()
    # Image loading
    newimage = np.array(my_img.convert('RGB'))  # Convert the image into RGB form
    img = cv2.cvtColor(newimage, 1)  # cvtColor()
    # Store the height, width and number of color channels of the image
    height, width, channels = img.shape


    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

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
                # object detetcted
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Adjust            confidence         threshold and NMS(Non - Maximum        Suppression) threshold

    score_threshold = st.sidebar.slider("Confidence_threshold", 0.00, 1.00, 0.5, 0.01)
    # nms_threshold = st.sidebar.slider("NMS_threshold", 0.00, 1.00, 0.4, 0.01)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    items = []  # Array to store label of detected object(s)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, label+" "+str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)

            items.append(label)  # Add the output label of bounded object

    # Display the  detected   objects  with anchor boxes (object_detection() continued)

    st.text("")  # preformatted and fixed-width text
    column2.subheader("Output image")  # Title on top of the output image
    st.text("")
    # Plot the output image with detected objects using matplotlib
    plt.figure(figsize=(15, 15))
    plt.imshow(img)  # show the figure
    column2.pyplot(use_column_width=True)  # actual plotting
    if len(indexes) > 1:
        # Text to be printed if output image has multiple detected objects
        st.success("Found {} Objects - {}".format(len(indexes), [item for
                                                                 item in set(items)]))

    else:
        # Text to be printed if output image has a single detected object
        st.success("Found {} Object - {}".format(len(indexes), [item for
                                                                item in set(items)]))


    rows = []
    for i in range(len(items)):
        #rows.append([items[i], items[i].count(items[i])])
        rows.append([items[i], items.count(items[i])])

    df = pd.DataFrame(rows, columns=["Class", "Count"]).drop_duplicates()
    st.dataframe(df)

def web_cam():

    # Display subheading on top of input image

    column1 = st.beta_columns(1)
    # st.text("")  # streamlit.text() writes preformatted and fixed-width text
    # Display the input image using matplotlib
    plt.figure(figsize=(20, 20))
    # plt.imshow(my_video)

    #st.video(my_video)
    start_button = st.button("Start")

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

        cap = cv2.VideoCapture(0)

        font = cv2.FONT_HERSHEY_PLAIN
        starting_time = time.time()
        frame_id = 0
        frameST = st.empty()
        while True:
            _, frame = cap.read()
            frame_id += 1
            height, width, channels = frame.shape[:3]

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

            cv2.imshow("Video ", frame)
            frameST.image(frame, channels="BGR")

            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
