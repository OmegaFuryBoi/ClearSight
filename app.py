import streamlit as st

from clearsight.loader import decode_image, limit_size, to_rgb

st.set_page_config(page_title="ClearSight", layout="wide")
st.title("ClearSight")
st.caption("Dual-mode real-time facial enhancement")

uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

img = limit_size(decode_image(uploaded.getvalue()))
h, w = img.shape[:2]

st.image(to_rgb(img), caption=f"Original ({w} x {h})")