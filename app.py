import streamlit as st
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import numpy as np

st.set_page_config(
    page_title="SmartAnnotate",
    page_icon="🖼️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #F8F7FC;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #24213A;
    margin-bottom: 0;
}

.subtitle {
    font-size: 17px;
    color: #6C5CE7;
    margin-top: -5px;
}

.developer {
    color: #77728D;
    font-size: 14px;
    font-weight: 500;
}

.section-title {
    color: #24213A;
    font-size: 24px;
    font-weight: 700;
}

[data-testid="stFileUploader"] {
    background-color: white;
    border: 2px dashed #B8B0F5;
    border-radius: 14px;
    padding: 12px;
}

div.stButton > button {
    border-radius: 9px;
    font-weight: 600;
    border: 1px solid #6C5CE7;
}

div.stButton > button:hover {
    border-color: #6C5CE7;
    color: #6C5CE7;
}

.sidebar-title {
    color: #24213A;
    font-size: 25px;
    font-weight: 800;
}

.sidebar-text {
    color: #77728D;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #8B879B;
    font-size: 13px;
    padding: 25px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">SmartAnnotate</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Image Annotation & Quality Control</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="developer">Developed by Mudassir Khan</div>',
    unsafe_allow_html=True
)

st.markdown("---")

st.sidebar.markdown(
    '<div class="sidebar-title">SmartAnnotate</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="sidebar-text">AI-Powered Image Annotation & Quality Control</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.subheader("Workflow")
st.sidebar.write("✓ Upload Images")
st.sidebar.write("✓ AI Detection")
st.sidebar.write("✓ Quality Check")
st.sidebar.write("✓ Human Review")
st.sidebar.write("✓ Correction")

st.sidebar.markdown("---")

st.sidebar.info(
    "YOLO-based person detection with human verification."
)

model = YOLO("yolo11n.pt")

output_folder = Path("output")
output_folder.mkdir(exist_ok=True)

if "decisions" not in st.session_state:
    st.session_state.decisions = {}

if "corrections" not in st.session_state:
    st.session_state.corrections = {}

st.markdown(
    '<div class="section-title">Upload Images</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload one or more images to start AI-powered annotation."
)

uploaded_files = st.file_uploader(
    "Select images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:

    total_images = len(uploaded_files)
    total_persons = 0

    for uploaded_file in uploaded_files:

        image = Image.open(uploaded_file).convert("RGB")

        results = model.predict(
            source=image,
            classes=[0],
            conf=0.25
        )

        result = results[0]
        person_count = len(result.boxes)
        total_persons += person_count

        annotated_image = result.plot()
        image_key = uploaded_file.name

        st.divider()

        st.markdown(
            f"### 📄 {uploaded_file.name}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Image**")
            st.image(image, width=300)

        with col2:
            st.markdown("**AI Annotation**")
            st.image(annotated_image, width=300)

        st.markdown(
            '<div class="section-title">Detection Results</div>',
            unsafe_allow_html=True
        )

        d1, d2 = st.columns(2)

        with d1:
            st.metric(
                "Persons Detected",
                person_count
            )

        with d2:

            if person_count > 0:

                confidence = max(
                    float(box.conf[0])
                    for box in result.boxes
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}"
                )

            else:

                st.metric(
                    "Confidence",
                    "N/A"
                )

        if person_count == 0:

            st.warning(
                "No person detected."
            )

        else:

            for i, box in enumerate(result.boxes):

                confidence = float(
                    box.conf[0]
                )

                if confidence >= 0.50:

                    st.success(
                        f"Person {i + 1} — OK — Confidence: {confidence:.2f}"
                    )

                else:

                    st.warning(
                        f"Person {i + 1} — REVIEW — Confidence: {confidence:.2f}"
                    )

        st.markdown(
            '<div class="section-title">Human Review</div>',
            unsafe_allow_html=True
        )

        decision = st.session_state.decisions.get(
            image_key
        )

        b1, b2 = st.columns(2)

        with b1:

            if st.button(
                "✓ Approve",
                key="approve_" + image_key
            ):

                st.session_state.decisions[
                    image_key
                ] = "Approved"

                decision = "Approved"

        with b2:

            if st.button(
                "⚠ Needs Review",
                key="review_" + image_key
            ):

                st.session_state.decisions[
                    image_key
                ] = "Needs Review"

                decision = "Needs Review"

        if decision == "Approved":

            st.success(
                "Human Review: Approved"
            )

        if decision == "Needs Review":

            st.warning(
                "Human Review: Needs Review"
            )

            st.markdown(
                "### Correct Annotation"
            )

            st.write(
                "Draw a bounding box around the person if the AI annotation is incorrect."
            )

            canvas_width = 700

            canvas_height = int(
                image.height
                * canvas_width
                / image.width
            )

            canvas_image = image.resize(
                (
                    canvas_width,
                    canvas_height
                )
            )

            canvas_result = st_canvas(
                fill_color="rgba(108, 92, 231, 0.15)",
                stroke_width=2,
                stroke_color="#6C5CE7",
                background_image=canvas_image,
                drawing_mode="rect",
                width=canvas_width,
                height=canvas_height,
                key="canvas_" + image_key
            )

            if st.button(
                "Save Correction",
                key="save_" + image_key
            ):

                if canvas_result.json_data:

                    objects = canvas_result.json_data[
                        "objects"
                    ]

                    if len(objects) > 0:

                        corrected_image = np.array(
                            canvas_image
                        ).copy()

                        for obj in objects:

                            if obj["type"] == "rect":

                                left = int(
                                    obj["left"]
                                )

                                top = int(
                                    obj["top"]
                                )

                                width = int(
                                    obj["width"]
                                )

                                height = int(
                                    obj["height"]
                                )

                                corrected_image[
                                    top:top + height,
                                    left:left + 2
                                ] = [108, 92, 231]

                                corrected_image[
                                    top:top + height,
                                    left + width - 2:left + width
                                ] = [108, 92, 231]

                                corrected_image[
                                    top:top + 2,
                                    left:left + width
                                ] = [108, 92, 231]

                                corrected_image[
                                    top + height - 2:top + height,
                                    left:left + width
                                ] = [108, 92, 231]

                        save_path = (
                            output_folder
                            / (
                                "corrected_"
                                + uploaded_file.name
                            )
                        )

                        Image.fromarray(
                            corrected_image
                        ).save(save_path)

                        st.session_state.corrections[
                            image_key
                        ] = "Corrected"

                        st.success(
                            "Corrected annotation saved successfully."
                        )

                    else:

                        st.warning(
                            "Please draw a bounding box first."
                        )

    st.divider()

    st.markdown(
        '<div class="section-title">Project Summary</div>',
        unsafe_allow_html=True
    )

    approved = 0
    reviewed = 0

    for value in st.session_state.decisions.values():

        if value == "Approved":
            approved += 1

        if value == "Needs Review":
            reviewed += 1

    corrected = len(
        st.session_state.corrections
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Images Processed",
            total_images
        )

    with c2:
        st.metric(
            "Persons Detected",
            total_persons
        )

    with c3:
        st.metric(
            "Approved",
            approved
        )

    with c4:
        st.metric(
            "Corrected",
            corrected
        )

    st.success(
        "Annotation quality checking completed."
    )

else:

    st.info(
        "Upload one or more images above to start."
    )

st.markdown(
    '<div class="footer">SmartAnnotate • AI-Powered Image Annotation & Quality Control • Developed by Mudassir Khan</div>',
    unsafe_allow_html=True
)