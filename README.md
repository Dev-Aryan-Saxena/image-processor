# Lumina CV: Spatial Enhancement & Geometrical Transformations

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lumina-cv.streamlit.app/)
&nbsp;&nbsp;**Live Demo:** [lumina-cv.streamlit.app](https://lumina-cv.streamlit.app/)

Lumina CV is an interactive, real-time computer vision application built in Python using Streamlit, OpenCV, and Matplotlib. It allows users to upload custom images or use a default calibration pattern, adjust processing parameters through interactive sliders, and see side-by-side comparisons of the original and processed images along with dynamic RGB/grayscale histograms and execution speeds.

---

## 🚀 Key Features

### 1. Spatial-Domain Enhancement (Point Processing)
Direct pixel-level mapping functions ($s = T(r)$) to improve image quality:
*   **Brightness Adjustment**: Add/subtract intensity offset constant values ($s = r + \beta$).
*   **Contrast Adjustment**: Scale intensity values using multiplication gain factors ($s = \alpha \cdot r$).
*   **Image Negative**: Inverse intensity mapping ($s = 255 - r$) to highlight dark features.
*   **Log Transformation**: Dynamic range compression ($s = c \cdot \ln(1 + r)$) to expand dark regions.
*   **Power-Law (Gamma) Transformation**: Non-linear gamma correction ($s = c \cdot r^\gamma$) to adjust mid-tones.
*   **Thresholding**: Map pixel values to binary limits (black and white) based on threshold limit $T$.
*   **Contrast Stretching**: Linearly stretch narrow input intensity ranges to fill the full $[0, 255]$ range.

### 2. Geometrical Transformations
Coordinate mappings to translate, shear, scale, or warp image shapes:
*   **Translation**: Shift image coordinates horizontally and vertically.
*   **Scaling**: Resize image dimensions with adjustable interpolation methods (Nearest Neighbor, Bilinear, Bicubic).
*   **Rotation**: Rotate the image by custom degrees around any pivot point (e.g. image center).
*   **Reflections**:
    *   *Horizontal*: Flips image horizontally.
    *   *Vertical*: Flips image vertically.
    *   *Origin*: Flips image both horizontally and vertically ($180^\circ$ rotation).
*   **Shearing**: Slant the image along the X-axis or Y-axis by a shearing factor.
*   **Affine Transformation**: Warp the image using three control points (Top-Left, Top-Right, Bottom-Left offsets) displaying the resulting $2 \times 3$ transformation matrix in real-time.

---

## 🛠️ Issues Encountered & Resolution

During development, the following runtime issue was encountered:

### ⚠️ ModuleNotFoundError: No module named 'matplotlib'
*   **Symptom**: When launching the application (`app.py`), the program crashed during initialization at:
    ```python
    import matplotlib.pyplot as plt
    ```
    Outputting a `ModuleNotFoundError` because `matplotlib` was missing from the virtual environment packages.
*   **Root Cause**: Matplotlib was not included in the initial installation command, which only installed `streamlit`, `opencv-python`, `numpy`, and `pillow`.
*   **Resolution**: 
    1. Activated the virtual environment and installed the package:
       ```bash
       ./venv/bin/pip install matplotlib
       ```
    2. Gracefully terminated the old server process.
    3. Restarted the Streamlit server daemon:
       ```bash
       ./venv/bin/streamlit run app.py --server.port=8501 --server.headless=true
       ```
    4. Verified successful compilation and loading.

---

## 🔧 Installation & Setup

### Requirements
*   Python 3.10+
*   pip

### Step-by-step Setup
1.  **Clone this repository**:
    ```bash
    git clone <repository-url>
    cd image-processor
    ```

2.  **Set up Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install streamlit opencv-python numpy pillow matplotlib
    ```

4.  **Run the application**:
    ```bash
    streamlit run app.py
    ```
    The application will open automatically in your default browser at `http://localhost:8501`.

---

## 👤 Developer
Built by Dev-Aryan-Saxena (contact.aryan.dev@gmail.com).
