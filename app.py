import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import io

# Set page configuration with a premium icon and responsive layout
st.set_page_config(
    page_title="Lumina CV - Image Processing Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for premium styling, dark mode accents, and clean layout
st.markdown("""
<style>
    /* Main body background styling */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .app-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card design for parameter control and info */
    .control-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
    }
    
    /* Custom divider */
    .custom-hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, rgba(59, 130, 246, 0), rgba(59, 130, 246, 0.75), rgba(59, 130, 246, 0));
        margin: 2rem 0;
    }
    
    /* Image display styling */
    .image-container {
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 6px;
        background-color: #182235;
        text-align: center;
    }
    
    /* Metrics box */
    .metric-box {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 10px 15px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- HELPER FUNCTIONS -----------------

def load_image(source, uploaded_file=None):
    """Loads image from uploader or default calibration path."""
    if source == "Upload Image" and uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            img_np = np.array(image)
            # Ensure 3 channels
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            return img_np
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
            return None
    else:
        # Load default image
        try:
            img = cv2.imread("default_input.jpg")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb
        except Exception as e:
            st.error("Default calibration image not found. Please upload an image.")
            return None

def plot_comparison_histograms(img_orig, img_proc):
    """Generates a clean side-by-side histogram for comparison using Matplotlib."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), facecolor='none')
    
    for a in ax:
        a.set_facecolor('none')
        a.tick_params(colors='#94a3b8', labelsize=8)
        a.xaxis.label.set_color('#94a3b8')
        a.yaxis.label.set_color('#94a3b8')
        for spine in a.spines.values():
            spine.set_color('#334155')
            
    colors = ('#ef4444', '#22c55e', '#3b82f6')  # Beautiful RGB colors
    
    # Original Image Histogram
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_orig], [i], None, [256], [0, 256])
        ax[0].plot(hist, color=color, alpha=0.8, linewidth=1.5)
        ax[0].fill_between(range(256), hist.flatten(), color=color, alpha=0.08)
    ax[0].set_title("Original Histogram", color='#f8fafc', fontsize=10, pad=10)
    ax[0].set_xlim([0, 256])
    ax[0].grid(True, color='#334155', linestyle='--', alpha=0.3)
    
    # Processed Image Histogram
    # Check if processed is grayscale
    is_gray = len(img_proc.shape) == 2 or (len(img_proc.shape) == 3 and img_proc.shape[2] == 1)
    if is_gray:
        hist = cv2.calcHist([img_proc], [0], None, [256], [0, 256])
        ax[1].plot(hist, color='#cbd5e1', alpha=0.8, linewidth=1.5)
        ax[1].fill_between(range(256), hist.flatten(), color='#cbd5e1', alpha=0.08)
    else:
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img_proc], [i], None, [256], [0, 256])
            ax[1].plot(hist, color=color, alpha=0.8, linewidth=1.5)
            ax[1].fill_between(range(256), hist.flatten(), color=color, alpha=0.08)
            
    ax[1].set_title("Processed Histogram", color='#f8fafc', fontsize=10, pad=10)
    ax[1].set_xlim([0, 256])
    ax[1].grid(True, color='#334155', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

# ----------------- SIDEBAR INTERFACE -----------------

st.sidebar.markdown('<div style="text-align: center; margin-top:-20px;"><span style="font-size:2.5rem;">🎨</span></div>', unsafe_allow_html=True)
st.sidebar.markdown('<h2 style="text-align:center; color:#f8fafc; font-weight:700; margin-bottom:5px;">Lumina CV Studio</h2>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="text-align:center; color:#94a3b8; font-size:0.9rem; margin-bottom:20px;">Image Enhancement & Warp Sandbox</p>', unsafe_allow_html=True)

# 1. Source Image Selection
st.sidebar.subheader("📥 Input Source")
input_source = st.sidebar.radio("Choose Input Image:", ("Calibration Image (Default)", "Upload Image"))
uploaded_file = None
if input_source == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# 2. Main Studio Mode
st.sidebar.subheader("🛠️ Sandbox Mode")
mode = st.sidebar.selectbox("Select Operation Category:", ("Spatial-Domain Enhancement", "Geometrical Transformations"))

# Dynamic technique loading based on category
if mode == "Spatial-Domain Enhancement":
    technique = st.sidebar.selectbox(
        "Select Technique:",
        (
            "Brightness Adjustment",
            "Contrast Adjustment",
            "Image Negative",
            "Log Transformation",
            "Power-Law (Gamma) Transformation",
            "Thresholding",
            "Contrast Stretching"
        )
    )
else:
    technique = st.sidebar.selectbox(
        "Select Technique:",
        (
            "Translation",
            "Scaling",
            "Rotation",
            "Horizontal Reflection",
            "Vertical Reflection",
            "Reflection about Origin",
            "Shearing (X-Axis)",
            "Shearing (Y-Axis)",
            "Affine Transformation"
        )
    )

# 3. Parameters configuration
st.sidebar.subheader("🎛️ Parameters")

# Initialize parameters dictionary
params = {}

if mode == "Spatial-Domain Enhancement":
    if technique == "Brightness Adjustment":
        params["beta"] = st.sidebar.slider("Brightness Offset (β)", -255, 255, 40, step=1, help="Add or subtract intensity constant value to adjust brightness.")
    
    elif technique == "Contrast Adjustment":
        params["alpha"] = st.sidebar.slider("Contrast Gain (α)", 0.0, 3.0, 1.5, step=0.05, help="Multiply pixel values by gain factor to scale contrast.")
        
    elif technique == "Image Negative":
        st.sidebar.info("Image negative has no free parameters. It maps input intensities linearly to their opposite: s = 255 - r.")
        
    elif technique == "Log Transformation":
        params["c_factor"] = st.sidebar.slider("Scale Multiplier (c_factor)", 0.1, 2.5, 1.0, step=0.1, help="Adjust scale constant scaling factor of the natural log curve.")
        
    elif technique == "Power-Law (Gamma) Transformation":
        params["gamma"] = st.sidebar.slider("Gamma Exponent (γ)", 0.05, 5.0, 0.5, step=0.05, help="Gamma < 1 stretches dark tones (brightens). Gamma > 1 compresses dark tones (darkens).")
        params["c_factor"] = st.sidebar.slider("Gain (c)", 0.1, 2.0, 1.0, step=0.1, help="Constant scale factor applied post gamma calculation.")
        
    elif technique == "Thresholding":
        params["threshold_mode"] = st.sidebar.selectbox("Thresholding Mode:", ("Grayscale (Standard)", "RGB Channel-wise"))
        params["threshold_val"] = st.sidebar.slider("Threshold Value (T)", 0, 255, 127, step=1, help="Pixels >= T will be set to max value (255), otherwise 0.")
        
    elif technique == "Contrast Stretching":
        st.sidebar.markdown("**Source / Input Limits**")
        use_actual = st.sidebar.checkbox("Auto-detect Input Limits (r_min, r_max)", value=True, help="Set input range r_min and r_max to actual minimum and maximum image intensities.")
        
        # Load image initially to get actual min/max for sliders
        initial_img = load_image(input_source, uploaded_file)
        if initial_img is not None:
            actual_min, actual_max = int(np.min(initial_img)), int(np.max(initial_img))
        else:
            actual_min, actual_max = 0, 255
            
        if use_actual:
            params["r_min"] = actual_min
            params["r_max"] = actual_max
            st.sidebar.caption(f"Detected: r_min={actual_min}, r_max={actual_max}")
        else:
            params["r_min"] = st.sidebar.slider("Input Min (r_min)", 0, 255, actual_min, step=1)
            params["r_max"] = st.sidebar.slider("Input Max (r_max)", 0, 255, actual_max, step=1)
            
        st.sidebar.markdown("**Destination / Output Limits**")
        params["s_min"] = st.sidebar.slider("Output Min (s_min)", 0, 255, 0, step=1)
        params["s_max"] = st.sidebar.slider("Output Max (s_max)", 0, 255, 255, step=1)

else:  # Geometrical Transformations
    # Background fill color option
    bg_choice = st.sidebar.selectbox("Background Fill Color:", ("Black", "White", "Gray", "Custom"))
    if bg_choice == "Black":
        params["bg_color"] = (0, 0, 0)
    elif bg_choice == "White":
        params["bg_color"] = (255, 255, 255)
    elif bg_choice == "Gray":
        params["bg_color"] = (128, 128, 128)
    else:
        bg_col_picker = st.sidebar.color_picker("Pick Fill Color:", "#000000")
        # Convert hex to RGB tuple
        r_col = int(bg_col_picker[1:3], 16)
        g_col = int(bg_col_picker[3:5], 16)
        b_col = int(bg_col_picker[5:7], 16)
        params["bg_color"] = (r_col, g_col, b_col)
        
    if technique == "Translation":
        params["tx"] = st.sidebar.slider("Horizontal Translation (t_x)", -300, 300, 50, step=5, help="Shift image horizontally in pixels.")
        params["ty"] = st.sidebar.slider("Vertical Translation (t_y)", -300, 300, 30, step=5, help="Shift image vertically in pixels.")
        
    elif technique == "Scaling":
        params["sx"] = st.sidebar.slider("Horizontal Scale (s_x)", 0.1, 3.0, 1.2, step=0.1, help="Scale factor along width.")
        params["sy"] = st.sidebar.slider("Vertical Scale (s_y)", 0.1, 3.0, 1.2, step=0.1, help="Scale factor along height.")
        params["interp"] = st.sidebar.selectbox(
            "Interpolation Method:",
            ("Bilinear (Smooth)", "Nearest Neighbor (Pixelated)", "Bicubic (High Quality)"),
            help="Mathematical approach to calculate new pixel intensities."
        )
        
    elif technique == "Rotation":
        params["angle"] = st.sidebar.slider("Rotation Angle (θ)", -180.0, 180.0, 30.0, step=1.0, help="Rotation degrees (positive counter-clockwise).")
        params["scale"] = st.sidebar.slider("Pre-scaling factor", 0.2, 2.0, 1.0, step=0.1, help="Additional scaling factor centered at rotation origin.")
        
        # Center coordinates selection
        initial_img = load_image(input_source, uploaded_file)
        if initial_img is not None:
            h, w = initial_img.shape[:2]
        else:
            w, h = 600, 400
        
        use_center = st.sidebar.checkbox("Rotate around Image Center", value=True)
        if use_center:
            params["xc"] = w / 2.0
            params["yc"] = h / 2.0
            st.sidebar.caption(f"Pivot: ({int(w/2)}, {int(h/2)})")
        else:
            params["xc"] = st.sidebar.slider("Pivot X (x_c)", 0.0, float(w), float(w)/2.0, step=1.0)
            params["yc"] = st.sidebar.slider("Pivot Y (y_c)", 0.0, float(h), float(h)/2.0, step=1.0)
            
    elif technique in ("Horizontal Reflection", "Vertical Reflection", "Reflection about Origin"):
        st.sidebar.info(f"{technique} has no parameters. It flips the image coordinates across axis limits.")
        
    elif technique == "Shearing (X-Axis)":
        params["kx"] = st.sidebar.slider("Shearing Factor (k_x)", -2.0, 2.0, 0.3, step=0.05, help="Displace coordinate x proportional to y coordinates.")
        
    elif technique == "Shearing (Y-Axis)":
        params["ky"] = st.sidebar.slider("Shearing Factor (k_y)", -2.0, 2.0, 0.3, step=0.05, help="Displace coordinate y proportional to x coordinates.")
        
    elif technique == "Affine Transformation":
        st.sidebar.markdown("**Point warping controls**")
        st.sidebar.write("Map 3 standard corners to new coordinates:")
        st.sidebar.markdown("*1. Top-Left Corner (0, 0) Shift:*")
        params["dx1"] = st.sidebar.slider("Shift TL - X", -150, 150, 20, step=1)
        params["dy1"] = st.sidebar.slider("Shift TL - Y", -150, 150, 15, step=1)
        
        st.sidebar.markdown("*2. Top-Right Corner (W-1, 0) Shift:*")
        params["dx2"] = st.sidebar.slider("Shift TR - X", -150, 150, -30, step=1)
        params["dy2"] = st.sidebar.slider("Shift TR - Y", -150, 150, 40, step=1)
        
        st.sidebar.markdown("*3. Bottom-Left Corner (0, H-1) Shift:*")
        params["dx3"] = st.sidebar.slider("Shift BL - X", -150, 150, 30, step=1)
        params["dy3"] = st.sidebar.slider("Shift BL - Y", -150, 150, -20, step=1)


# ----------------- MAIN LAYOUT -----------------

st.markdown('<div class="app-header">Lumina CV Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Interactive Point Processing & Geometrical Transformation Sandbox</div>', unsafe_allow_html=True)

# Load selected image
img_orig = load_image(input_source, uploaded_file)

if img_orig is None:
    st.warning("Please upload an image or ensure default calibration image is available.")
else:
    rows, cols, channels = img_orig.shape
    
    # ----------------- PROCESS IMAGE -----------------
    
    start_time = time.perf_counter()
    
    # Apply selected technique
    img_proc = None
    
    if mode == "Spatial-Domain Enhancement":
        if technique == "Brightness Adjustment":
            beta = params["beta"]
            img_proc = np.clip(img_orig.astype(np.int32) + beta, 0, 255).astype(np.uint8)
            
        elif technique == "Contrast Adjustment":
            alpha = params["alpha"]
            img_proc = np.clip(img_orig.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            
        elif technique == "Image Negative":
            img_proc = (255 - img_orig).astype(np.uint8)
            
        elif technique == "Log Transformation":
            c_factor = params["c_factor"]
            r = img_orig.astype(np.float32)
            c_base = 255.0 / np.log(1.0 + np.max(r)) if np.max(r) > 0 else 1.0
            img_proc = (c_base * c_factor * np.log(1.0 + r))
            img_proc = np.clip(img_proc, 0, 255).astype(np.uint8)
            
        elif technique == "Power-Law (Gamma) Transformation":
            gamma = params["gamma"]
            c_factor = params["c_factor"]
            r = img_orig.astype(np.float32) / 255.0
            img_proc = 255.0 * c_factor * (r ** gamma)
            img_proc = np.clip(img_proc, 0, 255).astype(np.uint8)
            
        elif technique == "Thresholding":
            threshold_val = params["threshold_val"]
            if params["threshold_mode"] == "Grayscale (Standard)":
                gray = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
                img_proc = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            else:
                img_proc = np.where(img_orig >= threshold_val, 255, 0).astype(np.uint8)
                
        elif technique == "Contrast Stretching":
            r_min = params["r_min"]
            r_max = params["r_max"]
            s_min = params["s_min"]
            s_max = params["s_max"]
            
            r = img_orig.astype(np.float32)
            denom = (r_max - r_min) if r_max != r_min else 1.0
            stretched = (r - r_min) * ((s_max - s_min) / denom) + s_min
            img_proc = np.clip(stretched, 0, 255).astype(np.uint8)
            
    else:  # Geometrical Transformations
        bg_color = params["bg_color"]
        
        if technique == "Translation":
            tx, ty = params["tx"], params["ty"]
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img_proc = cv2.warpAffine(img_orig, M, (cols, rows), borderValue=bg_color)
            
        elif technique == "Scaling":
            sx, sy = params["sx"], params["sy"]
            interp_str = params["interp"]
            if "Nearest" in interp_str:
                interp = cv2.INTER_NEAREST
            elif "Bicubic" in interp_str:
                interp = cv2.INTER_CUBIC
            else:
                interp = cv2.INTER_LINEAR
                
            new_w = int(cols * sx)
            new_h = int(rows * sy)
            img_proc = cv2.resize(img_orig, (new_w, new_h), interpolation=interp)
            
        elif technique == "Rotation":
            angle = params["angle"]
            scale = params["scale"]
            xc, yc = params["xc"], params["yc"]
            M = cv2.getRotationMatrix2D((xc, yc), angle, scale)
            img_proc = cv2.warpAffine(img_orig, M, (cols, rows), borderValue=bg_color)
            
        elif technique == "Horizontal Reflection":
            img_proc = cv2.flip(img_orig, 1)
            
        elif technique == "Vertical Reflection":
            img_proc = cv2.flip(img_orig, 0)
            
        elif technique == "Reflection about Origin":
            img_proc = cv2.flip(img_orig, -1)
            
        elif technique == "Shearing (X-Axis)":
            kx = params["kx"]
            M = np.float32([[1, kx, 0], [0, 1, 0]])
            img_proc = cv2.warpAffine(img_orig, M, (cols, rows), borderValue=bg_color)
            
        elif technique == "Shearing (Y-Axis)":
            ky = params["ky"]
            M = np.float32([[1, 0, 0], [ky, 1, 0]])
            img_proc = cv2.warpAffine(img_orig, M, (cols, rows), borderValue=bg_color)
            
        elif technique == "Affine Transformation":
            dx1, dy1 = params["dx1"], params["dy1"]
            dx2, dy2 = params["dx2"], params["dy2"]
            dx3, dy3 = params["dx3"], params["dy3"]
            
            src_pts = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
            dst_pts = np.float32([
                [0 + dx1, 0 + dy1],
                [cols-1 + dx2, dy2],
                [dx3, rows-1 + dy3]
            ])
            M = cv2.getAffineTransform(src_pts, dst_pts)
            img_proc = cv2.warpAffine(img_orig, M, (cols, rows), borderValue=bg_color)
            
    end_time = time.perf_counter()
    processing_time_ms = (end_time - start_time) * 1000.0
    
    # ----------------- DISPLAY SIDE-BY-SIDE -----------------
    
    tab_studio, tab_math = st.tabs(["✨ Studio Workbench", "📚 Mathematical Reference"])
    
    with tab_studio:
        # Columns for metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{cols} × {rows}</div>
                <div class="metric-label">Original Shape (W × H)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            proc_h, proc_w = img_proc.shape[:2]
            st.markdown(f"""
            <div class="metric-box" style="border-left-color: #8b5cf6;">
                <div class="metric-value">{proc_w} × {proc_h}</div>
                <div class="metric-label">Processed Shape (W × H)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="metric-box" style="border-left-color: #ec4899;">
                <div class="metric-value">{processing_time_ms:.2f} ms</div>
                <div class="metric-label">Execution Time</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            st.markdown(f"""
            <div class="metric-box" style="border-left-color: #10b981;">
                <div class="metric-value">{img_proc.dtype}</div>
                <div class="metric-label">Data Format</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        
        # Original and Processed display
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown('<h3 style="color:#f8fafc; font-size:1.2rem; font-weight:600; margin-bottom:10px;">Original Image</h3>', unsafe_allow_html=True)
            st.image(img_orig, use_container_width=True)
            
        with col_right:
            st.markdown(f'<h3 style="color:#f8fafc; font-size:1.2rem; font-weight:600; margin-bottom:10px;">Processed: {technique}</h3>', unsafe_allow_html=True)
            st.image(img_proc, use_container_width=True)
            
        # Download and Info
        st.write("")
        
        # Save and Download action
        pil_proc = Image.fromarray(img_proc)
        buf = io.BytesIO()
        # Handle format based on extension or convert to PNG
        pil_proc.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        d_col1, d_col2 = st.columns([1, 4])
        with d_col1:
            st.download_button(
                label="📥 Download Result",
                data=byte_im,
                file_name=f"processed_{technique.lower().replace(' ', '_')}.png",
                mime="image/png"
            )
            
        # Dynamic matrix display for geometrical mappings
        if mode == "Geometrical Transformations":
            with d_col2:
                if technique in ("Translation", "Rotation", "Shearing (X-Axis)", "Shearing (Y-Axis)", "Affine Transformation"):
                    st.write("**Current Transformation Matrix $M$ ($2 \\times 3$):**")
                    # Construct nice display table or latex matrix
                    st.latex(rf"""
                    M = \begin{bmatrix} 
                    {M[0,0]:.4f} & {M[0,1]:.4f} & {M[0,2]:.2f} \\
                    {M[1,0]:.4f} & {M[1,1]:.4f} & {M[1,2]:.2f}
                    \end{bmatrix}
                    """)
                    
        st.markdown('<hr class="custom-hr" />', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#f8fafc; font-size:1.3rem; font-weight:700; margin-bottom:15px;">📊 Intensity Distribution Comparison</h3>', unsafe_allow_html=True)
        
        # Plot and render histograms
        hist_fig = plot_comparison_histograms(img_orig, img_proc)
        st.pyplot(hist_fig)
        
    with tab_math:
        st.markdown('<h2 style="color:#f8fafc; font-size:1.6rem; font-weight:700; margin-bottom:10px;">Theory & Math Formulas</h2>', unsafe_allow_html=True)
        
        st.subheader("Spatial-Domain Enhancements")
        
        st.markdown("""
        Spatial domain point processing modifies the intensity of individual pixels directly using a mapping function:
        $$s = T(r)$$
        where $r$ is the input intensity, $s$ is the output intensity, and $T$ is the transformation operator.
        """)
        
        # Render markdown and LaTeX equations depending on which one was selected, or show all
        math_col1, math_col2 = st.columns(2)
        
        with math_col1:
            st.markdown("### Point Processing Formulas")
            st.markdown(r"""
            *   **Brightness Adjustment**:
                $$s = r + \beta$$
                $\beta > 0$ increases brightness; $\beta < 0$ decreases it. Values are clipped to $[0, 255]$.
            
            *   **Contrast Adjustment**:
                $$s = \alpha \cdot r$$
                $\alpha > 1$ stretches contrast; $\alpha < 1$ compresses it.
            
            *   **Image Negative**:
                $$s = (L - 1) - r$$
                Reverses the intensity levels (for an 8-bit image, $L = 256$, so $s = 255 - r$).
                
            *   **Log Transformation**:
                $$s = c \cdot \ln(1 + r)$$
                Compresses dynamic range of images with high variation in pixel values (e.g. Fourier transform spectra). Expansion of dark levels, compression of bright levels.
                
            *   **Power-Law (Gamma) Transformation**:
                $$s = c \cdot r^\gamma$$
                Maps narrow range of dark inputs into wider range of outputs for $\gamma < 1$, and opposite for $\gamma > 1$. Essential for monitor screen response correction.
                
            *   **Thresholding**:
                $$s = \begin{cases} L - 1 & \text{if } r \geq T \\ 0 & \text{otherwise} \end{cases}$$
                Produces a binary (black-and-white) image based on a threshold limit $T$.
                
            *   **Contrast Stretching**:
                $$s = (r - r_{min}) \cdot \frac{s_{max} - s_{min}}{r_{max} - r_{min}} + s_{min}$$
                Linearly expands the narrow intensity range $[r_{min}, r_{max}]$ to fill a wider output range $[s_{min}, s_{max}]$.
            """)
            
        with math_col2:
            st.markdown("### Geometrical Transformation Formulas")
            st.markdown(r"""
            Geometrical transformations map spatial coordinates $(x,y)$ to $(x',y')$. They are represented in homogeneous coordinates as matrix multiplications:
            
            *   **Translation**:
                $$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
            
            *   **Scaling**:
                $$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
            
            *   **Rotation**:
                Rotating by angle $\theta$ around a pivot center $(x_c, y_c)$:
                $$x' = (x - x_c)\cos\theta - (y - y_c)\sin\theta + x_c$$
                $$y' = (x - x_c)\sin\theta + (y - y_c)\cos\theta + y_c$$
            
            *   **Reflections (Flips)**:
                *   Horizontal: $x' = W - 1 - x, \ y' = y$
                *   Vertical: $x' = x, \ y' = H - 1 - y$
                *   Origin: $x' = W - 1 - x, \ y' = H - 1 - y$
                
            *   **Shearing**:
                *   X-Axis Shear: $x' = x + k_x \cdot y, \ y' = y$
                *   Y-Axis Shear: $x' = x, \ y' = y + k_y \cdot x$
                
            *   **Affine Transformation**:
                A general transformation preserving collinearity and ratios of distances. Defined by 6 degrees of freedom mapping three source points to three destination points:
                $$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}$$
            """)
            
        st.markdown('<hr style="border-color:#334155;"/>', unsafe_allow_html=True)
        st.markdown("**Developer Note:** Built with Streamlit, OpenCV, and Matplotlib. All algorithms are computed in real-time on CPU. Image interpolation uses Bilinear/Bicubic filters as selected.")

# ----------------- FOOTER -----------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#64748b; font-size:0.8rem;">Lumina CV Studio • Designed with Advanced Agentic Coding Tools</p>', unsafe_allow_html=True)
