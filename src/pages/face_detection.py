import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
from datetime import datetime


CSV_FILE = "registered_students.csv"


def detect_faces(image):
    img = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for i, (x, y, w, h) in enumerate(faces, start=1):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

        cv2.putText(
            img,
            f"Face {i}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return img, faces


def create_csv_if_not_exists():
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=["Name", "Course", "Roll No", "Registered At"])
        df.to_csv(CSV_FILE, index=False)


def load_registered_students():
    create_csv_if_not_exists()
    return pd.read_csv(CSV_FILE)


def save_student(name, course, roll_no):
    create_csv_if_not_exists()

    df = pd.read_csv(CSV_FILE)

    new_data = {
        "Name": name.strip(),
        "Course": course.strip(),
        "Roll No": str(roll_no).strip(),
        "Registered At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)


def check_already_registered(roll_no):
    df = load_registered_students()

    if df.empty:
        return False, None

    roll_no = str(roll_no).strip()
    df["Roll No"] = df["Roll No"].astype(str).str.strip()

    matched = df[df["Roll No"] == roll_no]

    if not matched.empty:
        return True, matched.iloc[0]

    return False, None


def show_registered_box(name, course, roll_no, status):
    st.markdown(f"""
    <div style="
        background-color:#d1fae5;
        padding:20px;
        border-radius:12px;
        border:2px solid #10b981;
        color:#064e3b;
        font-size:18px;
    ">
        <b>Name:</b> {name}<br>
        <b>Course:</b> {course}<br>
        <b>Roll No:</b> {roll_no}<br>
        <b>Status:</b> {status}
    </div>
    """, unsafe_allow_html=True)


def show():
    st.title("📷 Face Detection + Student Registration")
    st.markdown("### Detect face and register student details")

    st.info("First detect a face, then enter student details.")

    if "camera_started" not in st.session_state:
        st.session_state.camera_started = False

    mode = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

    image = None

    if mode == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload face image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    else:
        if not st.session_state.camera_started:
            if st.button("📸 Start Camera"):
                st.session_state.camera_started = True
                st.rerun()

        if st.session_state.camera_started:
            camera_file = st.camera_input("Take a clear front-facing photo")

            if st.button("❌ Stop Camera"):
                st.session_state.camera_started = False
                st.rerun()

            if camera_file is not None:
                image = Image.open(camera_file)

    if image is not None:
        result_img, faces = detect_faces(image)

        st.image(
            result_img,
            caption=f"Faces detected: {len(faces)}",
            use_container_width=True
        )

        st.metric("Total Faces Detected", len(faces))

        if len(faces) == 0:
            st.warning("""
            No face detected. Try:
            - clear front-facing photo
            - better lighting
            - closer camera distance
            """)
            return

        st.success("✅ Face detected. You can now register.")

        st.divider()

        st.subheader("📝 Student Registration Form")

        name = st.text_input("Enter Name")
        course = st.text_input("Enter Course")
        roll_no = st.text_input("Enter Roll Number")

        # Check immediately after roll number is entered
        if roll_no.strip():
            already_registered, student = check_already_registered(roll_no)

            if already_registered:
                st.success("✅ You are already registered.")

                show_registered_box(
                    student["Name"],
                    student["Course"],
                    student["Roll No"],
                    "Already Registered"
                )

                return

        # Register button appears only when roll number is not already registered
        if st.button("Register Student", use_container_width=True):
            if not name.strip() or not course.strip() or not roll_no.strip():
                st.warning("Please fill all fields.")
                return

            save_student(name, course, roll_no)

            st.success("🎉 Registration Successful")

            show_registered_box(
                name,
                course,
                roll_no,
                "Registered Successfully"
            )

    st.divider()

    with st.expander("📄 View Registered Students"):
        df = load_registered_students()
        st.dataframe(df, use_container_width=True)