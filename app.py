import streamlit as st

from clearsight.loader import decode_image, limit_size, to_rgb
from clearsight.degrade import degrade

st.set_page_config(page_title="ClearSight", layout="wide")
st.title("ClearSight")
st.caption("Dual-mode real-time facial enhancement")

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Degradation")
    blur = st.slider("Blur", 0, 10, 0)
    scale = st.slider("Resolution drop (x)", 1, 8, 1)
    gamma = st.slider("Darkness (gamma)", 1.0, 4.0, 1.0, 0.1)
    noise_sigma = st.slider("Grain (Gaussian noise)", 0, 50, 0)
    sp_amount = st.slider("Salt & pepper noise", 0.0, 0.10, 0.0, 0.005)

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