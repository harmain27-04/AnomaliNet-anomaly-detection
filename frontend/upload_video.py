import streamlit as st
import os

st.title("Upload Surveillance Video")

video = st.file_uploader(
    "Upload Video",
    type=["mp4","avi","mov"]
)

if video:

    os.makedirs(
        "uploaded_videos",
        exist_ok=True
    )

    save_path = os.path.join(
        "uploaded_videos",
        video.name
    )

    with open(save_path, "wb") as f:

        f.write(video.read())

    st.success(
        "Video Uploaded"
    )