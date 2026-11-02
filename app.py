import streamlit as st

from clearsight.loader import decode_image, limit_size, to_rgb
from clearsight.degrade import degrade, PRESETS

st.set_page_config(page_title="ClearSight", layout="wide")
st.title("ClearSight")
st.caption("Dual-mode real-time facial enhancement")

# ---------- Sidebar controls ----------
def apply_preset():
    """Copy the chosen preset's values into the sliders."""
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

# ---------- Input ----------
uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

img = limit_size(decode_image(uploaded.getvalue()))
h, w = img.shape[:2]

# ---------- Processing ----------
degraded = degrade(img, blur, scale, gamma, noise_sigma, sp_amount)

# ---------- Display ----------
col1, col2 = st.columns(2)
col1.image(to_rgb(img), caption=f"Original ({w} x {h})")
col2.image(to_rgb(degraded), caption="Degraded")