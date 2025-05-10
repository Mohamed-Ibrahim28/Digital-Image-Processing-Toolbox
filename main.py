import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from FN import (
    apply_edge_detection, apply_blur, apply_sharpen, apply_hist_eq, apply_rotation,
    cv2_to_pil,
    point_add, point_subtract, point_divide, point_complement,
    change_light, change_red, swap_rg, eliminate_red,
    hist_stretch, hist_equalize,
    avg_filter, lap_filter, max_filter, min_filter, median_filter, mode_filter,
    add_salt_pepper, remove_sp_avg, remove_sp_median, remove_sp_outlier,
    add_gauss, remove_gauss_avg,
    thresh_basic, thresh_auto, thresh_adapt,
    sobel,
    dilate, erode, opening, boundary_internal, boundary_external, boundary_gradient
)

st.title("🖼️ Digital Image Processing Toolbox")
st.write("Created by Mohamed Ibrahim Mohamed")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(cv2_to_pil(img), caption="Input Image", width=300)

    topics = {
        "Point operation": {
            "Addition": point_add,
            "Subtraction": point_subtract,
            "Division": point_divide,
            "Complement": point_complement
        },
        "Color image operation": {
            "Change image lighting color": change_light,
            "Change red": change_red,
            "Swap from R to G": swap_rg,
            "Eliminate Red": eliminate_red
        },
        "Image histogram": {
            "Histogram Stretch (gray)": hist_stretch,
            "Histogram Equalization (gray)": hist_equalize
        },
        "Neighborhood processing": {
            "Average filter": avg_filter,
            "Laplacian filter": lap_filter,
            "Max filter": max_filter,
            "Min filter": min_filter,
            "Median filter": median_filter,
            "Mode filter": mode_filter
        },
        "Image Restoration": {
            "Add Salt & Pepper noise": add_salt_pepper,
            "Remove S&P (Avg filter)": remove_sp_avg,
            "Remove S&P (Median filter)": remove_sp_median,
            "Remove S&P (Outlier method)": remove_sp_outlier,
            "Add Gaussian noise": add_gauss,
            "Remove Gaussian (Avg filter)": remove_gauss_avg
        },
        "Image segmentation": {
            "Basic Global Thresholding": thresh_basic,
            "Automatic Thresholding": thresh_auto,
            "Adaptive Thresholding": thresh_adapt
        },
        "Edge Detection": {
            "Sobel detector": sobel,
            "Canny edge": apply_edge_detection
        },
        "Mathematical Morphology": {
            "Image dilation": dilate,
            "Image erosion": erode,
            "Image opening": opening,
            "Internal boundary": boundary_internal,
            "External boundary": boundary_external,
            "Morphological gradient": boundary_gradient
        },
        "Advanced Filters": {
            "Gaussian Blur": apply_blur,
            "Sharpen": apply_sharpen
        },
        "Transformations": {
            "Histogram Equalization": apply_hist_eq,
            "Rotate Image 45°": apply_rotation
        }
    }

    topic = st.selectbox("Select Topic", list(topics.keys()))
    operation = st.selectbox("Select Operation", list(topics[topic].keys()))

    if st.button("Apply Operation"):
        try:
            result_img = topics[topic][operation](img)
            result_pil = cv2_to_pil(result_img)

            st.image(result_pil, caption=f"Result: {operation}", width=300)

            # Display side-by-side comparison
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[0].set_title("Input")
            axs[0].axis('off')
            axs[1].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            axs[1].set_title("Result")
            axs[1].axis('off')
            st.pyplot(fig)

            # Download button
            from io import BytesIO
            buf = BytesIO()
            result_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button("Download Result", byte_im, file_name="result.png", mime="image/png")

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
