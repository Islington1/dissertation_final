import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def image_function(my_img):

    st.set_option('deprecation.showPyplotGlobalUse', False)

    column1, column2 = st.beta_columns(2)

    # Display subheading on top of input image
    column1.subheader("Input image")  # streamlit.subheader()

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








