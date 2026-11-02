import streamlit as st

from clearsight.loader import decode_image, limit_size, to_rgb
from clearsight.degrade import degrade, PRESETS
from clearsight.enhance import gamma_correct, equalize_global, equalize_clahe, denoise, sharpen

st.set_page_config(page_title="ClearSight", layout="wide")
st.title("ClearSight")
st.caption("Dual-mode real-time facial enhancement")

# ---------- Sidebar controls ----------
def apply_preset():
    for name, value in PRESETS[st.session_state.preset].items():
        st.session_state[name] = value

# First run only: start every slider at the "Clean" values
for name, value in PRESETS["Clean"].items():
    if name not in st.session_state:
        st.session_state[name] = value

with st.sidebar:
    st.header("Degradation")
    st.selectbox("Preset", list(PRESETS), key="preset", on_change=apply_preset)
    blur = st.slider("Blur", 0, 10, key="blur")
    scale = st.slider("Resolution drop (x)", 1, 8, key="scale")
    gamma = st.slider("Darkness (gamma)", 1.0, 4.0, step=0.1, key="gamma")
    noise_sigma = st.slider("Grain (Gaussian noise)", 0, 50, key="noise_sigma")
    sp_amount = st.slider("Salt & pepper noise", 0.0, 0.10, step=0.005, key="sp_amount")
    st.header("Enhancement")

    st.subheader("1. Noise removal")
    denoise_method = st.radio("Denoise filter", ["None", "Gaussian", "Median", "Bilateral"])
    strength = st.slider("Filter strength", 1, 5, 2) if denoise_method != "None" else 0
    denoise_first = st.checkbox("Denoise before brightening", value=True)

    st.subheader("2. Brightness & contrast")
    boost = st.slider("Brightness boost (gamma)", 1.0, 4.0, 1.0, 0.1)
    contrast = st.radio("Contrast", ["None", "Global equalization", "CLAHE"])
    if contrast == "CLAHE":
        clip = st.slider("CLAHE clip limit", 1.0, 5.0, 2.0, 0.5)
        tile = st.slider("CLAHE tile size", 2, 16, 8)

    st.subheader("3. Sharpening")
    sharpen_method = st.radio("Sharpen filter", ["None", "Unsharp mask", "Laplacian"])
    amount = st.slider("Sharpen amount", 0.1, 3.0, 1.0, 0.1) if sharpen_method != "None" else 0

# ---------- Input ----------
uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

img = limit_size(decode_image(uploaded.getvalue()))
h, w = img.shape[:2]

# ---------- Processing ----------
degraded = degrade(img, blur, scale, gamma, noise_sigma, sp_amount)

enhanced = degraded

if denoise_first:
    enhanced = denoise(enhanced, denoise_method, strength)

enhanced = gamma_correct(enhanced, 1 / boost)
if contrast == "Global equalization":
    enhanced = equalize_global(enhanced)
elif contrast == "CLAHE":
    enhanced = equalize_clahe(enhanced, clip, tile)

if not denoise_first:
    enhanced = denoise(enhanced, denoise_method, strength)

enhanced = sharpen(enhanced, sharpen_method, amount)

# ---------- Display ----------
col1, col2, col3 = st.columns(3)
col1.image(to_rgb(img), caption=f"Original ({w} x {h})")
col2.image(to_rgb(degraded), caption="Degraded")
col3.image(to_rgb(enhanced), caption="Enhanced")
