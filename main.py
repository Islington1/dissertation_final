from PIL import Image
from yolo_img_detection import *
from yolo_video_detection import *
from web_cam_detection import *
from yolo_custom_detection import *
from streamlitWebCam import *
from read_log import *


def main():

    model_selection = st.sidebar.selectbox(
        'Choose one of the data trained type',('Welcome', 'Yolo Pretrained Model', 'Yolo Custom Trained Model', 'Codes')
    )

    if model_selection == "Yolo Pretrained Model":

        selected_box = st.sidebar.selectbox(
            'Choose one of the following media format',
            ('About YOLO Dataset', 'Image Object Detection', 'Video', 'Web Cam')
        )

        if selected_box == 'About YOLO Dataset':
            about_yolo()

        if selected_box == 'Image Object Detection':
            object_detection()

        if selected_box == 'Video':
            video_detection()

        if selected_box == 'Web Cam':
            web_video_detection()

    elif model_selection == "Yolo Custom Trained Model":
        selected_box1 = st.sidebar.selectbox(
            'Choose one of the following option',
            ('About Custom Dataset', 'Activate Real time Detection')
        )

        if selected_box1 == 'About Custom Dataset':
            about_custom_dataset()

        if selected_box1 == 'Activate Real time Detection':
            custom_detection()

    elif model_selection == "Codes":

        selected_box2 = st.sidebar.selectbox(
            'Choose one of the following codes',
            ('Main', 'Image Detection', 'Video Detection', 'Web Cam', 'Custom Detection')
        )
        if selected_box2 == 'Main':
            print_log("main.py")

        if selected_box2 == 'Image Detection':
            print_log("yolo_img_detection.py")

        if selected_box2 == 'Video Detection':
            print_log("yolo_video_detection.py")

        if selected_box2 == 'Web Cam':
            print_log("streamlitWebCam.py")
        if selected_box2 == 'Custom Detection':
            print_log("yolo_custom_detection.py")

    elif model_selection == "Welcome":
        welcome()


def about_yolo():
    st.title('Introcution to pre trained models of YOLO')
    st.write('YOLO model has been trained on the MS COCO datasets for the thousands of iteration and hours of training.')
    st.write('Thus it provides us the different pretrained models for the multi class object detection.')
    st.subheader('YOLO is trained on the 80 classes')
    st.write('In this preoject, we have used yolo-tiny weights and configuration files provided by YOLO to detect objects.')


def about_custom_dataset():
    st.title('About YOLO Custom Training')
    st.write('For YOLO custom training, classess of Sign Languages were trained to detect the meaning of Sign.')
    st.header('Data Collection')
    st.write('Three different calsses of the Sign laguage were chosen for the custom training.')
    st.markdown('Hello, Thank You, I Love you')
    st.write('Images with respective to the classess were collected from the Google.')

    st.header('Data Labelling')
    st.write('Data labelling is the process of highlighting the data features, characteristics or classification '
             'which can be  analyzed for the patterns to help in the prediction and object detection process.')
    st.write('An open source labelling tool "labelImg" was used to label these collected images or data')

    st.header('Custom Training')
    st.write('Google Colab was used to train this custom model since it provides free GPU. The colab was connected to '
             'the google drive, backing up all training performed in the colab notebook')

def custom_detection():

    st.title('Real time custom trained for Sign language Detection')
    col1, col2, col3 = st.beta_columns(3)
    hello = Image.open("images/hello.jpg")
    col1.header("Hello")
    col1.image(hello, use_column_width=True)

    thankyou = Image.open("images/thankyou.jpg")
    col2.header("Thank You")
    col2.image(thankyou, use_column_width=True)

    iloveyou = Image.open("images/iloveyou.jpg")
    col3.header("I Love You")
    col3.image(iloveyou, use_column_width=True)

    custom_object_detection()


def welcome():
    st.title('Object Detection Using Deep Learning Algorithm')

    st.header('YOLO: You Only Look Once')

    st.write('In this app, we have used YOLOv3, a real-time object detection algorithm that uses a fully'
             ' convolutional network, along with a Darknet backbone, which performs feature extraction')

    st.header('A simple app that shows object detection on image, video and live video. ')

    st.subheader(' You can choose the options from the left.')
    st.write('See the sample of object detection in an Image')
    sampleImage = Image.open("images//sample_image.jpg")
    st.image(sampleImage)

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

    # If user selects 2nd option:
    if choice == "Upload Video of your choice":
        st.subheader("Input Video")
        video_file = st.file_uploader("Upload", type=['mp4'])

        if video_file is not None:  # if a file has been uploaded

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())

            # perform object detection on selected image
            video_function(tfile.name)

        # If user selects 1st option
    elif choice == "See an illustration":
        # display the example image
        video_function("images/test1.mp4")


def object_detection():

    st.header('Object Detection In an Image')

    st.write('Finding all the objects in an image and drawing the so-called bounding boxes around them, is the main motive. ')

    choice = st.radio("", ("See an illustration", "Choose an image of your choice"))

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

    web_cam1()


if __name__ == "__main__":
    main()
