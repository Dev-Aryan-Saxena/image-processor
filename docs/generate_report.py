import os
import shutil
import cv2
import numpy as np
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from datetime import datetime

# Initialize paths and settings
SCRATCH_DIR = "/home/saxena_ji/.gemini/antigravity-cli/scratch"
TEMP_DIR = os.path.join(SCRATCH_DIR, "temp_images_v3")
os.makedirs(TEMP_DIR, exist_ok=True)

# Path to original image in workspace
INPUT_IMAGE_PATH = "/home/saxena_ji/Projects/image-processor/default_input.jpg"

# Outputs directory in workspace
OUTPUT_DOCS_DIR = "/home/saxena_ji/Projects/image-processor/docs"
os.makedirs(OUTPUT_DOCS_DIR, exist_ok=True)

PDF_FILENAME = os.path.join(OUTPUT_DOCS_DIR, "Lumina_CV_Processing_Report.pdf")

# 1. Load default calibration image or create synthetic grid
img_bgr = cv2.imread(INPUT_IMAGE_PATH)
if img_bgr is None:
    print(f"Warning: {INPUT_IMAGE_PATH} not found, creating synthetic calibration image.")
    # Create a synthetic colorful test pattern
    img_bgr = np.zeros((400, 600, 3), dtype=np.uint8)
    # Background pattern
    for y in range(0, 400, 40):
        cv2.line(img_bgr, (0, y), (600, y), (80, 80, 80), 1)
    for x in range(0, 600, 40):
        cv2.line(img_bgr, (x, 0), (x, 400), (80, 80, 80), 1)
    # Add shapes
    cv2.rectangle(img_bgr, (50, 50), (250, 350), (200, 50, 50), -1)
    cv2.circle(img_bgr, (450, 200), 120, (50, 200, 50), -1)
    cv2.circle(img_bgr, (450, 200), 60, (50, 50, 200), -1)
    cv2.ellipse(img_bgr, (300, 200), (80, 40), 45, 0, 360, (200, 200, 50), -1)
    cv2.putText(img_bgr, "LUMINA CV", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)

img_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# Resize to moderate resolution for report layout
img_orig_resized = cv2.resize(img_orig, (360, 240), interpolation=cv2.INTER_AREA)

# Save original image to temp folder
orig_path = os.path.join(TEMP_DIR, "original.png")
PILImage.fromarray(img_orig_resized).save(orig_path)

# Definition of operations
operations = []

# --- SPATIAL ENHANCEMENT TECHNIQUES ---

def op_brightness(img):
    beta = 50
    return np.clip(img.astype(np.int32) + beta, 0, 255).astype(np.uint8)

operations.append({
    "title": "Brightness Adjustment",
    "category": "Spatial-Domain Enhancement",
    "formula": "s = r + &beta; (where &beta; = 50)",
    "desc": "Adds a constant brightness offset to each pixel intensity. Pixel values are clipped to the valid range [0, 255] to prevent overflow wraps.",
    "code": """beta = 50
img_proc = np.clip(img_orig.astype(np.int32) + beta, 0, 255).astype(np.uint8)""",
    "run_func": op_brightness
})

def op_contrast(img):
    alpha = 1.6
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

operations.append({
    "title": "Contrast Adjustment",
    "category": "Spatial-Domain Enhancement",
    "formula": "s = &alpha; &middot; r (where &alpha; = 1.6)",
    "desc": "Scales pixel intensities by a constant gain factor. Values greater than 1 stretch the image contrast, while values less than 1 compress it.",
    "code": """alpha = 1.6
img_proc = np.clip(img_orig.astype(np.float32) * alpha, 0, 255).astype(np.uint8)""",
    "run_func": op_contrast
})

def op_negative(img):
    return (255 - img).astype(np.uint8)

operations.append({
    "title": "Image Negative",
    "category": "Spatial-Domain Enhancement",
    "formula": "s = 255 - r",
    "desc": "Inverts the intensity levels of the image, transforming light pixels into dark and vice-versa, making structural details in dark regions easier to analyze.",
    "code": """img_proc = (255 - img_orig).astype(np.uint8)""",
    "run_func": op_negative
})

def op_log_trans(img):
    c_factor = 1.2
    r = img.astype(np.float32)
    c_base = 255.0 / np.log(1.0 + np.max(r)) if np.max(r) > 0 else 1.0
    return np.clip(c_base * c_factor * np.log(1.0 + r), 0, 255).astype(np.uint8)

operations.append({
    "title": "Log Transformation",
    "category": "Spatial-Domain Enhancement",
    "formula": "s = c &middot; ln(1 + r)",
    "desc": "Compresses the dynamic range of images containing extremely high intensity variations. It expands low-intensity (dark) values and compresses high-intensity (bright) values.",
    "code": """c_factor = 1.2
r = img_orig.astype(np.float32)
c_base = 255.0 / np.log(1.0 + np.max(r)) if np.max(r) > 0 else 1.0
img_proc = np.clip(c_base * c_factor * np.log(1.0 + r), 0, 255).astype(np.uint8)""",
    "run_func": op_log_trans
})

def op_gamma_trans(img):
    gamma = 0.4
    c_factor = 1.0
    r = img.astype(np.float32) / 255.0
    return np.clip(255.0 * c_factor * (r ** gamma), 0, 255).astype(np.uint8)

operations.append({
    "title": "Power-Law (Gamma) Transformation",
    "category": "Spatial-Domain Enhancement",
    "formula": "s = c &middot; r<sup>&gamma;</sup> (where &gamma; = 0.4, c = 1.0)",
    "desc": "Applies non-linear correction to mid-tones. A fractional gamma (&gamma; &lt; 1) maps a narrower range of dark inputs into a wider range of outputs (brightens), while &gamma; &gt; 1 compresses dark tones (darkens).",
    "code": """gamma = 0.4
c_factor = 1.0
r = img_orig.astype(np.float32) / 255.0
img_proc = np.clip(255.0 * c_factor * (r ** gamma), 0, 255).astype(np.uint8)""",
    "run_func": op_gamma_trans
})

def op_threshold_gray(img):
    threshold_val = 120
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

operations.append({
    "title": "Thresholding (Grayscale)",
    "category": "Spatial-Domain Enhancement",
    "formula": "s = 255 if r<sub>gray</sub> &ge; T else 0 (where T = 120)",
    "desc": "Converts a grayscale version of the image into a binary black-and-white output. Intensity values greater than or equal to the threshold value T are set to 255 (white), others to 0 (black).",
    "code": """threshold_val = 120
gray = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
img_proc = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)""",
    "run_func": op_threshold_gray
})

def op_threshold_rgb(img):
    threshold_val = 120
    return np.where(img >= threshold_val, 255, 0).astype(np.uint8)

operations.append({
    "title": "Thresholding (RGB Channel-wise)",
    "category": "Spatial-Domain Enhancement",
    "formula": "s<sub>i</sub> = 255 if r<sub>i</sub> &ge; T else 0 (T = 120 for RGB)",
    "desc": "Performs binary thresholding independently on each color channel. This results in vibrant, highly saturated primary/secondary colors.",
    "code": """threshold_val = 120
img_proc = np.where(img_orig >= threshold_val, 255, 0).astype(np.uint8)""",
    "run_func": op_threshold_rgb
})

def op_contrast_stretching(img):
    r_min, r_max = int(np.min(img)), int(np.max(img))
    s_min, s_max = 0, 255
    r = img.astype(np.float32)
    denom = (r_max - r_min) if r_max != r_min else 1.0
    stretched = (r - r_min) * ((s_max - s_min) / denom) + s_min
    return np.clip(stretched, 0, 255).astype(np.uint8)

operations.append({
    "title": "Contrast Stretching",
    "category": "Spatial-Domain Enhancement",
    "formula": "s = (r - r<sub>min</sub>) &middot; (s<sub>max</sub> - s<sub>min</sub>)/(r<sub>max</sub> - r<sub>min</sub>) + s<sub>min</sub>",
    "desc": "Linearly maps narrow input intensity range [r_min, r_max] to fill the entire dynamic destination range [s_min, s_max] (usually [0, 255]). Useful for brightening low-contrast, washed-out images.",
    "code": """r_min, r_max = int(np.min(img_orig)), int(np.max(img_orig))
s_min, s_max = 0, 255
r = img_orig.astype(np.float32)
denom = (r_max - r_min) if r_max != r_min else 1.0
stretched = (r - r_min) * ((s_max - s_min) / denom) + s_min
img_proc = np.clip(stretched, 0, 255).astype(np.uint8)""",
    "run_func": op_contrast_stretching
})


# --- GEOMETRICAL TRANSFORMATION TECHNIQUES ---

def op_translation(img):
    tx, ty = 50, 30
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderValue=(30, 41, 59))  # Dark slate background fill

operations.append({
    "title": "Translation",
    "category": "Geometrical Transformations",
    "formula": "x' = x + t<sub>x</sub>, y' = y + t<sub>y</sub> (where t<sub>x</sub> = 50, t<sub>y</sub> = 30)",
    "desc": "Shifts the image spatial coordinates horizontally and vertically by specified pixel displacements. Empty boundary regions are filled with a default background color.",
    "code": """tx, ty = 50, 30
rows, cols = img_orig.shape[:2]
M = np.float32([[1, 0, tx], [0, 1, ty]])
img_proc = cv2.warpAffine(img_orig, M, (cols, rows), borderValue=(30, 41, 59))""",
    "run_func": op_translation
})

def op_scaling(img):
    sx, sy = 1.3, 1.3
    h, w = img.shape[:2]
    new_w, new_h = int(w * sx), int(h * sy)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

operations.append({
    "title": "Scaling (Bilinear)",
    "category": "Geometrical Transformations",
    "formula": "x' = s<sub>x</sub> &middot; x, y' = s<sub>y</sub> &middot; y (where s<sub>x</sub> = 1.3, s<sub>y</sub> = 1.3)",
    "desc": "Resizes the spatial footprint of the image. Standard bilinear interpolation is used to estimate intermediate pixel intensities, producing a smooth scaling result.",
    "code": """sx, sy = 1.3, 1.3
rows, cols = img_orig.shape[:2]
new_w, new_h = int(cols * sx), int(rows * sy)
img_proc = cv2.resize(img_orig, (new_w, new_h), interpolation=cv2.INTER_LINEAR)""",
    "run_func": op_scaling
})

def op_rotation(img):
    angle, scale = 25.0, 0.95
    h, w = img.shape[:2]
    xc, yc = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((xc, yc), angle, scale)
    return cv2.warpAffine(img, M, (w, h), borderValue=(30, 41, 59))

operations.append({
    "title": "Rotation",
    "category": "Geometrical Transformations",
    "formula": "Rotate by &theta; around pivot (x<sub>c</sub>, y<sub>c</sub>) (&theta; = 25&deg;, scale = 0.95)",
    "desc": "Rotates the coordinates by an angle theta around a central pivot point, optionally scaling the image dimensions relative to the center to keep it in frame.",
    "code": """angle, scale = 25.0, 0.95
rows, cols = img_orig.shape[:2]
xc, yc = cols / 2.0, rows / 2.0
M = cv2.getRotationMatrix2D((xc, yc), angle, scale)
img_proc = cv2.warpAffine(img_orig, M, (cols, rows), borderValue=(30, 41, 59))""",
    "run_func": op_rotation
})

def op_flip_horiz(img):
    return cv2.flip(img, 1)

operations.append({
    "title": "Horizontal Reflection",
    "category": "Geometrical Transformations",
    "formula": "x' = W - 1 - x, y' = y",
    "desc": "Flips the image along the vertical axis of symmetry (mirror effect), reversing the left and right layout of the image details.",
    "code": """img_proc = cv2.flip(img_orig, 1)""",
    "run_func": op_flip_horiz
})

def op_flip_vert(img):
    return cv2.flip(img, 0)

operations.append({
    "title": "Vertical Reflection",
    "category": "Geometrical Transformations",
    "formula": "x' = x, y' = H - 1 - y",
    "desc": "Flips the image upside down along the horizontal axis of symmetry.",
    "code": """img_proc = cv2.flip(img_orig, 0)""",
    "run_func": op_flip_vert
})

def op_flip_origin(img):
    return cv2.flip(img, -1)

operations.append({
    "title": "Reflection about Origin",
    "category": "Geometrical Transformations",
    "formula": "x' = W - 1 - x, y' = H - 1 - y (180&deg; rotation)",
    "desc": "Flips coordinates across both the horizontal and vertical axes of symmetry, equivalent to rotating the image by 180 degrees around its center.",
    "code": """img_proc = cv2.flip(img_orig, -1)""",
    "run_func": op_flip_origin
})

def op_shear_x(img):
    kx = 0.25
    h, w = img.shape[:2]
    M = np.float32([[1, kx, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderValue=(30, 41, 59))

operations.append({
    "title": "Shearing (X-Axis)",
    "category": "Geometrical Transformations",
    "formula": "x' = x + k<sub>x</sub> &middot; y, y' = y (where k<sub>x</sub> = 0.25)",
    "desc": "Displaces the horizontal coordinate of each pixel proportional to its vertical distance from the origin. This slants the image shapes sideways.",
    "code": """kx = 0.25
rows, cols = img_orig.shape[:2]
M = np.float32([[1, kx, 0], [0, 1, 0]])
img_proc = cv2.warpAffine(img_orig, M, (cols, rows), borderValue=(30, 41, 59))""",
    "run_func": op_shear_x
})

def op_shear_y(img):
    ky = 0.25
    h, w = img.shape[:2]
    M = np.float32([[1, 0, 0], [ky, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderValue=(30, 41, 59))

operations.append({
    "title": "Shearing (Y-Axis)",
    "category": "Geometrical Transformations",
    "formula": "x' = x, y' = y + k<sub>y</sub> &middot; x (where k<sub>y</sub> = 0.25)",
    "desc": "Displaces the vertical coordinate of each pixel proportional to its horizontal distance from the origin. This slants the image shapes vertically.",
    "code": """ky = 0.25
rows, cols = img_orig.shape[:2]
M = np.float32([[1, 0, 0], [ky, 1, 0]])
img_proc = cv2.warpAffine(img_orig, M, (cols, rows), borderValue=(30, 41, 59))""",
    "run_func": op_shear_y
})

def op_affine(img):
    dx1, dy1 = 20, 15
    dx2, dy2 = -30, 40
    dx3, dy3 = 30, -20
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1]])
    dst_pts = np.float32([
        [0 + dx1, 0 + dy1],
        [w-1 + dx2, dy2],
        [dx3, h-1 + dy3]
    ])
    M = cv2.getAffineTransform(src_pts, dst_pts)
    return cv2.warpAffine(img, M, (w, h), borderValue=(30, 41, 59))

operations.append({
    "title": "Affine Transformation",
    "category": "Geometrical Transformations",
    "formula": "Warp using 3 non-collinear matching control points",
    "desc": "Calculates a 2x3 transformation matrix from three matching control points. Preserves collinearity and straight lines, but alters angles, distances, and shapes.",
    "code": """dx1, dy1 = 20, 15
dx2, dy2 = -30, 40
dx3, dy3 = 30, -20
rows, cols = img_orig.shape[:2]
src_pts = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
dst_pts = np.float32([
    [0 + dx1, 0 + dy1],
    [cols-1 + dx2, dy2],
    [dx3, rows-1 + dy3]
])
M = cv2.getAffineTransform(src_pts, dst_pts)
img_proc = cv2.warpAffine(img_orig, M, (cols, rows), borderValue=(30, 41, 59))""",
    "run_func": op_affine
})

# --- RUN PROCESSING AND SAVE IMAGES ---
print("Running operations...")
for idx, op in enumerate(operations):
    # Run the function
    img_out = op["run_func"](img_orig_resized)
    
    # Save the output image
    out_name = f"processed_{idx}.png"
    out_path = os.path.join(TEMP_DIR, out_name)
    PILImage.fromarray(img_out).save(out_path)
    
    # Store reference paths and shapes
    op["orig_path"] = orig_path
    op["proc_path"] = out_path
    op["orig_shape"] = img_orig_resized.shape
    op["proc_shape"] = img_out.shape
    print(f"Processed: {op['title']}")

# --- REPORTLAB PDF GENERATION ---
print("Generating PDF...")

# Custom NumberedCanvas for professional headers and footers
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Header
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#1e3a8a"))
        self.drawString(36, 812, "LUMINA CV STUDIO: PROCESSING REPORT")
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawRightString(559, 812, "Image Enhancement & Transformations")
        
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.75)
        self.line(36, 805, 559, 805)
        
        # Footer
        self.line(36, 45, 559, 45)
        self.setFont("Helvetica", 8)
        self.drawString(36, 32, f"Date: {datetime.now().strftime('%B %d, %Y')}")
        self.drawRightString(559, 32, f"Page {self._pageNumber} of {page_count}")
        
        self.restoreState()

# Custom Styles Setup
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=20,
    leading=24,
    textColor=colors.HexColor('#1e3a8a'),
    spaceAfter=15
)

h1_style = ParagraphStyle(
    'H1Header',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=14,
    leading=18,
    textColor=colors.HexColor('#1e3a8a'),
    spaceBefore=15,
    spaceAfter=10,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    'H2Header',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=11,
    leading=14,
    textColor=colors.HexColor('#0f172a'),
    spaceBefore=12,
    spaceAfter=6,
    keepWithNext=True
)

body_style = ParagraphStyle(
    'BodyTextCustom',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9,
    leading=13,
    textColor=colors.HexColor('#334155'),
    spaceAfter=6
)

code_style = ParagraphStyle(
    'MonospaceCode',
    parent=styles['Normal'],
    fontName='Courier',
    fontSize=7.2,
    leading=9.2,
    textColor=colors.HexColor('#0f172a')
)

def make_code_block(code_text):
    escaped = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>').replace(' ', '&nbsp;')
    p = Paragraph(f"<font face='Courier'>{escaped}</font>", code_style)
    t = Table([[p]], colWidths=[523])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    return t

def make_image_table(input_path, output_path, input_shape, output_shape):
    def get_proportional_dims(shape, max_w=240, max_h=130):
        h, w = shape[:2]
        aspect = w / h
        new_w = max_w
        new_h = int(new_w / aspect)
        if new_h > max_h:
            new_h = max_h
            new_w = int(new_h * aspect)
        return new_w, new_h
        
    w_in, h_in = get_proportional_dims(input_shape)
    w_out, h_out = get_proportional_dims(output_shape)
    
    img_in = RLImage(input_path, width=w_in, height=h_in)
    img_out = RLImage(output_path, width=w_out, height=h_out)
    
    title_style = ParagraphStyle(
        'ImgTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        alignment=1,
        textColor=colors.HexColor('#475569')
    )
    
    lbl_in = Paragraph("Original (Input) Image", title_style)
    lbl_out = Paragraph("Processed (Output) Image", title_style)
    
    data = [
        [lbl_in, lbl_out],
        [img_in, img_out]
    ]
    
    t = Table(data, colWidths=[261, 261])
    t.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,0), 3),
        ('TOPPADDING', (0,1), (-1,1), 3),
        ('BOX', (0,1), (0,1), 0.75, colors.HexColor('#cbd5e1')),
        ('BOX', (1,1), (1,1), 0.75, colors.HexColor('#cbd5e1')),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    return t

story = []

# --- Document Header (Immediately starts on Page 1) ---
story.append(Paragraph("Lumina CV Studio: Reference Report", title_style))
story.append(Paragraph("This reference document lists the mathematical operations and implementation logic for spatial enhancements and geometrical transformations supported in Lumina CV Studio.", body_style))
story.append(Spacer(1, 10))

# --- OPERATION DETAILS ---
current_category = ""

for idx, op in enumerate(operations):
    # Print Section Category Header if it changes
    if op["category"] != current_category:
        current_category = op["category"]
        story.append(Paragraph(f"{current_category}", h1_style))
        story.append(Spacer(1, 6))
        
    op_flowables = []
    
    # Sub-heading
    op_title = f"{idx + 1}. {op['title']}"
    op_flowables.append(Paragraph(op_title, h2_style))
    
    # Math & Description
    desc_html = f"<b>Formula:</b> <i>{op['formula']}</i><br/>"
    desc_html += f"<b>Description:</b> {op['desc']}"
    op_flowables.append(Paragraph(desc_html, body_style))
    op_flowables.append(Spacer(1, 4))
    
    # Code block
    op_flowables.append(make_code_block(op['code']))
    op_flowables.append(Spacer(1, 6))
    
    # Images comparison
    op_flowables.append(make_image_table(op["orig_path"], op["proc_path"], op["orig_shape"], op["proc_shape"]))
    op_flowables.append(Spacer(1, 15))
    
    # Append to story keeping them together
    story.append(KeepTogether(op_flowables))

# --- COMPLETE REFERENCE CODE SECTION ---
story.append(PageBreak())
story.append(Paragraph("Complete Reference Source Code", h1_style))
story.append(Paragraph("The following blocks compile the full standalone reference code for Lumina CV Studio's image manipulation algorithms. The script is free of UI/UX code and can be run independently.", body_style))
story.append(Spacer(1, 8))

# Define the modular code segments manually to avoid overflowing a single page/table
code_segments = [
    # Segment 1: Setup & Imports
    ("1. Setup & Utility Imports", """import cv2
import numpy as np"""),

    # Segment 2: Spatial point processing helpers
    ("2. Spatial Domain Point Processing Helpers", """def adjust_brightness(img, beta=50):
    \"\"\"Adds a constant brightness offset to each pixel intensity.
    Values are clipped to the valid range [0, 255] to prevent overflow.
    \"\"\"
    return np.clip(img.astype(np.int32) + beta, 0, 255).astype(np.uint8)

def adjust_contrast(img, alpha=1.6):
    \"\"\"Scales pixel intensities by a constant gain factor.
    alpha > 1 stretches contrast; alpha < 1 compresses it.
    \"\"\"
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

def image_negative(img):
    \"\"\"Inverts the intensity levels of the image (s = 255 - r).\"\"\"
    return (255 - img).astype(np.uint8)

def log_transformation(img, c_factor=1.2):
    \"\"\"Compresses dynamic range by expanding dark intensities 
    and compressing bright intensities.
    \"\"\"
    r = img.astype(np.float32)
    c_base = 255.0 / np.log(1.0 + np.max(r)) if np.max(r) > 0 else 1.0
    return np.clip(c_base * c_factor * np.log(1.0 + r), 0, 255).astype(np.uint8)"""),

    # Segment 3: More spatial helpers
    ("3. Advanced Spatial Enhancements", """def gamma_transformation(img, gamma=0.4, c=1.0):
    \"\"\"Applies power-law intensity mapping.
    gamma < 1 brightens mid-tones; gamma > 1 darkens mid-tones.
    \"\"\"
    r = img.astype(np.float32) / 255.0
    return np.clip(255.0 * c * (r ** gamma), 0, 255).astype(np.uint8)

def threshold_grayscale(img, threshold_val=120):
    \"\"\"Converts a grayscale version of the image into binary black-and-white.\"\"\"
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

def threshold_rgb(img, threshold_val=120):
    \"\"\"Performs binary thresholding independently on each color channel.\"\"\"
    return np.where(img >= threshold_val, 255, 0).astype(np.uint8)

def contrast_stretching(img, s_min=0, s_max=255):
    \"\"\"Linearly expands the narrow input dynamic range to fill [s_min, s_max].\"\"\"
    r_min, r_max = int(np.min(img)), int(np.max(img))
    r = img.astype(np.float32)
    denom = (r_max - r_min) if r_max != r_min else 1.0
    stretched = (r - r_min) * ((s_max - s_min) / denom) + s_min
    return np.clip(stretched, 0, 255).astype(np.uint8)"""),

    # Segment 4: Geometrical helpers part 1
    ("4. Geometrical Transformations (Affine & Scaling)", """def translate_image(img, tx=50, ty=30, bg_color=(30, 41, 59)):
    \"\"\"Shifts image spatial coordinates by tx horizontally and ty vertically.\"\"\"
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderValue=bg_color)

def scale_image(img, sx=1.3, sy=1.3):
    \"\"\"Resizes the width and height of the image using bilinear interpolation.\"\"\"
    h, w = img.shape[:2]
    new_w, new_h = int(w * sx), int(h * sy)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def rotate_image(img, angle=25.0, scale=0.95, bg_color=(30, 41, 59)):
    \"\"\"Rotates the image by angle degrees around its center point.\"\"\"
    h, w = img.shape[:2]
    xc, yc = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((xc, yc), angle, scale)
    return cv2.warpAffine(img, M, (w, h), borderValue=bg_color)"""),

    # Segment 5: Reflection and Shearing
    ("5. Reflections & Shearing", """def reflect_horizontal(img):
    \"\"\"Flips the image horizontally (mirror flip along vertical axis).\"\"\"
    return cv2.flip(img, 1)

def reflect_vertical(img):
    \"\"\"Flips the image vertically (upside-down flip along horizontal axis).\"\"\"
    return cv2.flip(img, 0)

def reflect_origin(img):
    \"\"\"Flips the image coordinates across both axes (180 degree rotation).\"\"\"
    return cv2.flip(img, -1)

def shear_x(img, kx=0.25, bg_color=(30, 41, 59)):
    \"\"\"Applies horizontal shear by factor kx.\"\"\"
    h, w = img.shape[:2]
    M = np.float32([[1, kx, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderValue=bg_color)

def shear_y(img, ky=0.25, bg_color=(30, 41, 59)):
    \"\"\"Applies vertical shear by factor ky.\"\"\"
    h, w = img.shape[:2]
    M = np.float32([[1, 0, 0], [ky, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderValue=bg_color)"""),

    # Segment 6: Affine warp & Test run
    ("6. Affine Warp & Standalone Main Execution Block", """def affine_transform(img, dx1=20, dy1=15, dx2=-30, dy2=40, dx3=30, dy3=-20, bg_color=(30, 41, 59)):
    \"\"\"Warps the image matching 3 control points (Top-Left, Top-Right, Bottom-Left).\"\"\"
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1]])
    dst_pts = np.float32([
        [0 + dx1, 0 + dy1],
        [w-1 + dx2, dy2],
        [dx3, h-1 + dy3]
    ])
    M = cv2.getAffineTransform(src_pts, dst_pts)
    return cv2.warpAffine(img, M, (w, h), borderValue=bg_color)

if __name__ == "__main__":
    # Standalone execution test with test image loading
    img_bgr = cv2.imread("default_input.jpg")
    if img_bgr is None:
        print("default_input.jpg not found. Creating dummy test image.")
        img_rgb = np.zeros((300, 450, 3), dtype=np.uint8)
        cv2.circle(img_rgb, (225, 150), 80, (255, 0, 0), -1)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
    print(f"Loaded source image shape: {img_rgb.shape}")
    brightened = adjust_brightness(img_rgb, beta=50)
    rotated = rotate_image(img_rgb, angle=30)
    print(f"Processed brightened shape: {brightened.shape}")
    print(f"Processed rotated shape: {rotated.shape}")""")
]

# Append code segments to the document
for label, code in code_segments:
    seg_flow = []
    seg_flow.append(Paragraph(f"<b>{label}</b>", h2_style))
    seg_flow.append(make_code_block(code))
    seg_flow.append(Spacer(1, 10))
    story.append(KeepTogether(seg_flow))

# Generate the PDF Document
doc = SimpleDocTemplate(
    PDF_FILENAME,
    pagesize=A4,
    leftMargin=36,
    rightMargin=36,
    topMargin=54,
    bottomMargin=54
)

doc.build(story, canvasmaker=NumberedCanvas)
print(f"PDF Generated successfully at {PDF_FILENAME}!")

# Copy self script to the workspace docs directory
shutil.copy(__file__, os.path.join(OUTPUT_DOCS_DIR, "generate_report.py"))
print(f"Copied script to {os.path.join(OUTPUT_DOCS_DIR, 'generate_report.py')}")

# Clean up temp images
shutil.rmtree(TEMP_DIR)
print("Temporary images deleted.")
