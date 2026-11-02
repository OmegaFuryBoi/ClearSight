import streamlit as st

from clearsight.loader import decode_image, limit_size, to_rgb
from clearsight.degrade import PRESETS
from clearsight.pipeline import MODES, run
from clearsight.segment import overlay_mask, draw_box, crop
from clearsight.upscale import enlarge
from clearsight.metrics import evaluate

st.set_page_config(page_title="ClearSight", layout="wide")
st.title("ClearSight")
st.caption("Dual-mode real-time facial enhancement")

# ---------- Sidebar controls ----------
def apply_preset():
    for name, value in PRESETS[st.session_state.preset].items():
        st.session_state[name] = value

def apply_mode():
    for name, value in MODES[st.session_state.mode].items():
        st.session_state[name] = value

for name, value in {**PRESETS["Clean"], **MODES["Forensic"]}.items():
    if name not in st.session_state:
        st.session_state[name] = value

with st.sidebar:
    st.header("Degradation")
    st.selectbox("Preset", list(PRESETS), key="preset", on_change=apply_preset)
    st.slider("Blur", 0, 10, key="blur")
    st.slider("Resolution drop (x)", 1, 8, key="scale")
    st.slider("Darkness (gamma)", 1.0, 4.0, step=0.1, key="gamma")
    st.slider("Grain (Gaussian noise)", 0, 50, key="noise_sigma")
    st.slider("Salt & pepper noise", 0.0, 0.10, step=0.005, key="sp_amount")

    st.header("Enhancement")
    st.selectbox("Mode", list(MODES), key="mode", on_change=apply_mode)

    st.subheader("1. Noise removal")
    st.radio("Denoise filter", ["None", "Gaussian", "Median", "Bilateral", "Auto"], key="denoise_method")
    if st.session_state.denoise_method != "None":
        st.slider("Filter strength", 1, 5, key="strength")
    st.checkbox("Denoise before brightening", key="denoise_first")

    st.subheader("2. Brightness & contrast")
    st.checkbox("Auto brightness", key="auto_boost")
    if not st.session_state.auto_boost:
        st.slider("Brightness boost (gamma)", 1.0, 4.0, step=0.1, key="boost")
    st.radio("Contrast", ["None", "Global equalization", "CLAHE"], key="contrast")
    if st.session_state.contrast == "CLAHE":
        st.slider("CLAHE clip limit", 1.0, 5.0, step=0.5, key="clip")
        st.slider("CLAHE tile size", 2, 16, key="tile")

    st.subheader("3. Sharpening")
    st.radio("Sharpen filter", ["None", "Unsharp mask", "Laplacian"], key="sharpen_method")
    if st.session_state.sharpen_method != "None":
        st.slider("Sharpen amount", 0.1, 3.0, step=0.1, key="amount")

    st.subheader("4. Skin smoothing")
    st.slider("Smoothing", 0.0, 1.0, step=0.1, key="smooth")

    st.header("Face enlargement")
    st.radio("Interpolation", ["Nearest", "Bilinear", "Bicubic"], key="up_method")
    st.checkbox("Progressive (2x steps)", key="progressive")
    st.slider("Post-enlarge sharpening", 0.0, 2.0, step=0.1, key="up_sharpen")

    st.header("Face detection")
    detect_on = st.radio("Detect face on", ["Uploaded image", "Degraded image"])
    seg_method = st.radio("Skin segmentation", ["YCrCb threshold", "K-means"])
    cr_range, cb_range, k = (133, 173), (77, 127), 3
    if seg_method == "YCrCb threshold":
        cr_range = st.slider("Cr range", 0, 255, (133, 173))
        cb_range = st.slider("Cb range", 0, 255, (77, 127))
    else:
        k = st.slider("Clusters (k)", 2, 6, 3)
    mask_clean = st.slider("Mask cleanup size", 1, 21, 7, 2)

deg = {name: st.session_state[name] for name in PRESETS["Clean"]}
enh = {name: st.session_state[name] for name in MODES["Forensic"]}
seg = dict(detect_on=detect_on, seg_method=seg_method, cr_range=cr_range, cb_range=cb_range, k=k, mask_clean=mask_clean)

# ---------- Input ----------
uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

img = limit_size(decode_image(uploaded.getvalue()))
h, w = img.shape[:2]

# ---------- Processing ----------
res = run(img, deg, enh, seg)
degraded, enhanced, box = res["degraded"], res["enhanced"], res["box"]

# ---------- Display ----------
col1, col2, col3 = st.columns(3)
col1.image(to_rgb(img), caption=f"Original ({w} x {h})")
col2.image(to_rgb(degraded), caption="Degraded")
col3.image(to_rgb(enhanced), caption=f"Enhanced ({st.session_state.mode} mode)")

st.subheader("Face detection")
col4, col5, col6 = st.columns(3)
col4.image(to_rgb(overlay_mask(res["detect_view"], res["mask"])), caption="Skin mask")
boxed = draw_box(res["detect_view"], res["person"], (255, 0, 0))
boxed = draw_box(boxed, res["head"], (0, 255, 255))
boxed = draw_box(boxed, box, (0, 255, 0))
col5.image(to_rgb(boxed), caption="HOG person → head → face" if res["person"] else "No person found: close-up mode")
if box:
    col6.image(to_rgb(crop(enhanced, box)), caption="Face crop")
else:
    col6.warning("No face found. Try adjusting the thresholds.")

if box:
    st.subheader("Face enlargement")
    th = 320
    tw = int(box[2] * th / box[3])
    c1, c2, c3 = st.columns(3)
    c1.image(to_rgb(enlarge(crop(degraded, box), (tw, th), "Nearest")), caption="Degraded (plain zoom)")
    c2.image(to_rgb(enlarge(crop(enhanced, box), (tw, th), enh["up_method"], enh["progressive"], enh["up_sharpen"])), caption=f"Enhanced + {enh['up_method']}")
    c3.image(to_rgb(enlarge(crop(img, box), (tw, th), "Bicubic")), caption="Original (reference)")

    st.subheader("Scores (face region)")
    rows = evaluate(crop(img, box), crop(degraded, box), crop(enhanced, box))
    before, after = rows[1], rows[2]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("PSNR (dB)", after["PSNR (dB)"], round(after["PSNR (dB)"] - before["PSNR (dB)"], 2))
    m2.metric("SSIM", after["SSIM"], round(after["SSIM"] - before["SSIM"], 3))
    m3.metric("LBP similarity", after["LBP similarity"], round(after["LBP similarity"] - before["LBP similarity"], 3))
    m4.metric("Sharpness", after["Sharpness"], round(after["Sharpness"] - before["Sharpness"], 1), delta_color="off")
    m5.metric("Edge density (%)", after["Edge density (%)"], round(after["Edge density (%)"] - before["Edge density (%)"], 2), delta_color="off")
    st.dataframe(rows, hide_index=True)
    st.caption("PSNR, SSIM and LBP compare each image with the original: higher means closer. Sharpness and edge density have no 'better' direction, since noise inflates both: closer to the Original row is better.")