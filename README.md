
# Image Processing Toolkit with OpenCV 🖼️🧠

This project is a comprehensive image processing toolkit built using Python, OpenCV, NumPy, and SciPy. It includes a wide variety of operations, filters, and effects commonly used in image analysis and manipulation.

---

## 🚀 Features

- ✅ Point operations: add, subtract, divide, complement
- ✅ Color manipulations: enhance brightness, adjust red channel, RGB swaps
- ✅ Histogram operations: stretching & equalization
- ✅ Spatial filters: average, median, laplacian, mode filter
- ✅ Noise handling: add/remove salt & pepper, gaussian noise
- ✅ Thresholding: basic, adaptive, and Otsu
- ✅ Edge detection: Sobel, Canny
- ✅ Morphological operations: dilation, erosion, opening, internal/external/gradient boundary
- ✅ Extra: blurring, sharpening, rotation, histogram equalization

---

## 🧰 Requirements

Make sure you have the following Python packages installed:

```bash
pip install opencv-python numpy scipy pillow
```

---

## 💻 How to Use

1. Clone the repository:
```bash
git clone https://github.com/Mohamed-Ibrahim28/Digital-Image-Processing-Toolbox.git
cd Digital-Image-Processing-Toolbox
```

2. Import the functions in your project:

```python
import cv2
from your_module_name import point_add, hist_equalize, apply_sharpen
img = cv2.imread("your_image.jpg")
result = point_add(img)
cv2.imshow("Result", result)
cv2.waitKey(0)
```

> Or integrate the functions in a Streamlit or GUI interface.

---

## 🧠 Modules Overview

| Category | Functions |
|---------|------------|
| **Point Ops** | `point_add`, `point_subtract`, `point_divide`, `point_complement` |
| **Color Ops** | `change_light`, `change_red`, `swap_rg`, `eliminate_red` |
| **Histogram** | `hist_stretch`, `hist_equalize` |
| **Spatial Filters** | `avg_filter`, `lap_filter`, `median_filter`, `mode_filter`, `min_filter`, `max_filter` |
| **Noise Ops** | `add_salt_pepper`, `add_gauss`, `remove_sp_avg`, `remove_sp_median`, `remove_sp_outlier`, `remove_gauss_avg` |
| **Thresholding** | `thresh_basic`, `thresh_auto`, `thresh_adapt` |
| **Edges** | `sobel`, `apply_edge_detection` |
| **Morphology** | `dilate`, `erode`, `opening`, `boundary_internal`, `boundary_external`, `boundary_gradient` |
| **Extras** | `apply_blur`, `apply_sharpen`, `apply_hist_eq`, `apply_rotation` |

---

## 🖼️ Sample Usage

```python
img = cv2.imread("cat.jpg")
blurred = apply_blur(img)
edges = apply_edge_detection(img)
cv2.imshow("Blurred", blurred)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
```

---

## 📄 License

This project is licensed under the MIT License.

---

## ✨ Author

Developed by **(Mohamed Ibrahim Mohamed)**
GitHub: [Mohamed-Ibrahim28](https://github.com/Mohamed-Ibrahim28)
