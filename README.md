#  Age & Gender Detection for Horror Roller Coaster

##  Project Description

This project implements a **real-time age and gender detection system** using a custom-trained machine learning model. The system is designed for a horror roller coaster ride where **visitors under 13 or over 60 are restricted** for safety.

If a restricted person is detected:
- They are **marked with a red rectangle** on the video feed.
- A message **"Not allowed"** is displayed.
- Their **age, gender, and time of detection** are logged to a CSV/Excel file.

---

## ✅ Features

- Real-time **age and gender prediction** from webcam.
- Custom-trained CNN model for robust performance.
- Safety filtering: denies access to underage and elderly.
- Visual feedback with colored bounding boxes (Red = Denied, Green = Allowed).
- Logs all entries to an Excel or CSV file with timestamps.

---

##  Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
