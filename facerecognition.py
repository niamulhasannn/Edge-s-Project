import cv2
import os

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Gender and Age model files
gender_proto = "deploy_gender.prototxt"
gender_model = "gender_net.caffemodel"
age_proto = "deploy_age.prototxt"
age_model = "age_net.caffemodel"

# Labels
GENDER_LIST = ["Male", "Female"]
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load models
if not (os.path.exists(gender_proto) and os.path.exists(gender_model) and os.path.exists(age_proto) and os.path.exists(age_model)):
    print("❌ One or more model files are missing.")
    exit()

gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)

# Start webcam
cap = cv2.VideoCapture(0)
print("🎥 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror mode
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w].copy()

        # Create blob for DNN
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.426337, 87.768914, 114.895847), swapRB=False)

        # Gender prediction
        gender_net.setInput(blob)
        gender = GENDER_LIST[gender_net.forward()[0].argmax()]

        # Age prediction
        age_net.setInput(blob)
        age = AGE_BUCKETS[age_net.forward()[0].argmax()]

        label = f"{gender}, {age}"

        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face, Age & Gender Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
