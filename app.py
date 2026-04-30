from flask import Flask, render_template, Response, redirect, url_for, session, request, jsonify
import cv2
import csv
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "secret"

# ====== FILES ======
ATTENDANCE_FILE = "attendance.csv"
STUDENTS_FILE = "students.csv"

# ====== LOAD MODEL ======
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# ====== LOAD LABELS ======
import pickle
with open("labels.pkl", "rb") as f:
    label_map = pickle.load(f)

# ====== CAMERA ======

camera = None
camera_running = False

camera = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ====== CREATE FILE ======
def create_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["roll_no", "date", "time", "status"])

create_file()

# ====== MARK ATTENDANCE ======
def mark_attendance(roll_no):
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    # check already marked
    with open(ATTENDANCE_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["roll_no"].strip() == roll_no.strip() and row["date"].strip() == today:
                return "Already Marked Today"
    # mark attendance
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([roll_no, today, current_time, "Present"])

    return "Marked"

# ====== VIDEO STREAM ======
def generate_frames():
    global camera_running, camera

    marked_today = set()

    while camera_running:

        if camera is None:
            break

        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            label, confidence = recognizer.predict(face)

            if confidence < 50:   # 🔥 strict accuracy
                roll_no = label_map[label]

                # 🔥 ALWAYS check attendance
                
                # Store in set to avoid repeated heavy processing
                if roll_no not in marked_today:
                    result = mark_attendance(roll_no)
                    marked_today.add(roll_no)

                else:
                    result = "Already Marked Today"
                # 🎯 Display status
                if result == "Already Marked Today":
                    text = "Already Marked"
                    color = (0, 0, 255)   # red
                else:
                    text = "Marked"
                    color = (0, 255, 0)   # green

                # Show roll number
                cv2.putText(frame, f"{roll_no}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Show status
                cv2.putText(frame, text, (x, y+h+25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Draw rectangle
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

            else:
                cv2.putText(frame, "Unknown", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

        # Convert frame to bytes
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # 🔥 RELEASE CAMERA AFTER STOP
    if camera:
        camera.release()
        camera = None

# ====== ROUTES ======
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        print("DEBUG:", username, password)  # 👈 check terminal

        if username == "admin" and password == "5123":
            session["admin"] = True
            return redirect(url_for("landing"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")
@app.route("/index")
def index():
    if not session.get("admin"):
        return redirect(url_for("login"))

    records = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE) as f:
            reader = csv.DictReader(f)
            records = list(reader)

    return render_template("index.html", records=records[::-1])

@app.route("/start_camera")
def start_camera():
    global camera, camera_running

    if not camera_running:
        camera = cv2.VideoCapture(0)
        camera_running = True
        print("Camera Started")

    return "started"

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/get_attendance")
def get_attendance():
    data = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE) as f:
            reader = csv.DictReader(f)
            data = list(reader)[::-1]
    return jsonify({"data": data})

@app.route("/stop_camera")
def stop_camera():
    global camera, camera_running

    camera_running = False

    if camera:
        camera.release()
        camera = None
        print("Camera Stopped")

    return "stopped"

@app.route("/landing")
def landing():

    if not session.get("admin"):
        return redirect(url_for("login"))

    total_students = 0
    present_today = 0

    today = datetime.now().strftime("%Y-%m-%d")

    students = set()
    present_students = set()

    # read students file
    if os.path.exists("students.csv"):
        with open("students.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                students.add(row["roll_no"].strip())

    total_students = len(students)

    # read attendance file
    if os.path.exists("attendance.csv"):
        with open("attendance.csv") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if row["date"].strip() == today:
                    present_students.add(row["roll_no"].strip())

    present_today = len(present_students)

    absent_today = total_students - present_today

    # percentage
    if total_students > 0:
        attendance_percent = round((present_today / total_students) * 100, 2)
    else:
        attendance_percent = 0

    return render_template(
        "landing.html",
        total_students=total_students,
        present_today=present_today,
        absent_today=absent_today,
        attendance_percent=attendance_percent
    )

@app.route("/report")
def report():

    if not session.get("admin"):
        return redirect(url_for("login"))

    report_data = {}
    all_dates = set()

    # Read attendance file
    if os.path.exists("attendance.csv"):
        with open("attendance.csv", "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                roll = row["roll_no"].strip()
                date = row["date"].strip()

                all_dates.add(date)

                if roll not in report_data:
                    report_data[roll] = {
                        "roll": roll,
                        "present": 0
                    }

                report_data[roll]["present"] += 1

    # total working days
    total_days = len(all_dates)

    # calculate percentage
    final_report = []
    for roll in report_data:
        present = report_data[roll]["present"]

        percent = round((present / total_days) * 100, 2) if total_days > 0 else 0

        final_report.append({
            "roll": roll,
            "present": present,
            "percent": percent
        })

    return render_template(
        "report.html",
        report=final_report,
        total_days=total_days
    )

@app.route("/student_login", methods=["GET", "POST"])
def student_login():
    if request.method == "POST":
        roll = request.form["roll_no"]
        password = request.form["password"]

        with open(STUDENTS_FILE) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["roll_no"] == roll and row["password"] == password:
                    session["student"] = True
                    session["roll"] = roll
                    return redirect(url_for("student_dashboard"))

    return render_template("student_login.html")

@app.route("/student_dashboard")
def student_dashboard():
    if not session.get("student"):
        return redirect(url_for("student_login"))

    roll = session["roll"]

    records = []
    student_dates = set()
    all_dates = set()

    if os.path.exists("attendance.csv"):
        with open("attendance.csv") as f:
            reader = csv.DictReader(f)

            for row in reader:
                all_dates.add(row["date"])

                if row["roll_no"].strip() == roll.strip():
                    records.append(row)
                    student_dates.add(row["date"])

    present_days = len(student_dates)

    # 🔥 INCLUDE TODAY EVEN IF NOT MARKED
    from datetime import datetime

    if all_dates:
        total_days = len(all_dates) 
    else:
        total_days = 0

    attendance_percent = round((present_days / total_days) * 100, 2) if total_days > 0 else 0

    print("DEBUG RECORDS:", records)   # 👈 CHECK

    return render_template(
        "student_dashboard.html",
        roll_no=roll,
        records=records[::-1],   # latest first
        total_present=present_days,
        total_days=total_days,
        attendance_percent=attendance_percent
    )

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)