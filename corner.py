
import streamlit as st
import cv2
import numpy as np

from PIL import Image

def perform_edge_detection(img, edge_type):
    if edge_type == 'Canny':
        edges = cv2.Canny(img, 50, 150)
    elif edge_type == 'LoG':
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Laplacian(img_blur, cv2.CV_64F)
        edges = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)
        edges = (edges * 255).astype(np.uint8)
    elif edge_type == 'DoG':
        img_blur1 = cv2.GaussianBlur(img, (5, 5), 0)
        img_blur2 = cv2.GaussianBlur(img, (9, 9), 0)
        edges = img_blur1 - img_blur2
    else:
        edges = img

    return edges

def crop_image(image):
    left = st.number_input("Left", min_value=0, max_value=image.shape[1], value=0)
    top = st.number_input("Top", min_value=0, max_value=image.shape[0], value=0)
    right = st.number_input("Right", min_value=left, max_value=image.shape[1], value=image.shape[1])
    bottom = st.number_input("Bottom", min_value=top, max_value=image.shape[0], value=image.shape[0])

    cropped_image = image[top:bottom, left:right, :]

    return cropped_image

def rotate_image(image):
    angle = st.slider("Rotation Angle", min_value=0, max_value=360, value=0)

    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))

    return rotated_image

def convert_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return grayscale_image

def scale_image(image):
    scale_factor = st.slider("Scale Factor", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    return scaled_image

def main():
    st.markdown(
        """
        <style>
            body {
                background-color: #282c35;
                color: #ffffff;
                font-family: 'Arial', sans-serif;
            }
            .main {
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                background-color: #373e49;
            }
            .sidebar .sidebar-content {
                background-color: #1f232a;
                color: #ffffff;
            }
            h1 {
                color: #61dafb;
            }
            .stSlider {
                color: #61dafb;
            }
            .stButton {
                background-color: #61dafb;
                color: #ffffff;
            }
            .stNumberInput input {
                color: #ffffff;
                background-color: #373e49;
                border: 1px solid #61dafb;
                border-radius: 4px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    main_container = st.container()
    main_container.markdown('<div class="main">', unsafe_allow_html=True)

    st.sidebar.title("ASSIGNMENT")
    selected_assignment = st.sidebar.radio("", ["Edge detection", "Image manipulation"])

    if selected_assignment == "Edge detection":
        st.title("Edge Detection App")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

            st.image(image, caption="Uploaded Image", use_column_width=True)

            edge_type = st.selectbox("Select Edge Detection Technique", ["Canny", "LoG", "DoG"])

            edges = perform_edge_detection(image, edge_type)

            st.image(edges, caption=f"{edge_type} Edge Detection", use_column_width=True)

    elif selected_assignment == "Image manipulation":
        st.title("Image Manipulation App")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), dtype=np.uint8), 1)
            st.image(image, caption="Original Image", channels="BGR", use_column_width=True)

            manipulation_type = st.selectbox("Select Image Manipulation Technique", ["Crop", "Rotate", "Grayscale", "Scale"])

            if manipulation_type == "Crop":
                cropped_image = crop_image(image)
                st.image(cropped_image, caption="Cropped Image", channels="BGR", use_column_width=True)

            elif manipulation_type == "Rotate":
                rotated_image = rotate_image(image)
                st.image(rotated_image, caption="Rotated Image", channels="BGR", use_column_width=True)

            elif manipulation_type == "Grayscale":
                grayscale_image = convert_to_grayscale(image)
                st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)

            elif manipulation_type == "Scale":
                scaled_image = scale_image(image)
                st.image(scaled_image, caption="Scaled Image", channels="BGR", use_column_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
