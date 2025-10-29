import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from metrics1 import record_metrics
import time

def run_analysis():
    MODEL_PATH = "anushka_part/microexpression_model.keras"
    IMG_SIZE = 75
    CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    model = load_model(MODEL_PATH)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [263, 387, 385, 362, 380, 373]
    LEFT_EYE_FULL = [33, 133, 159, 145, 153, 154, 155, 133]
    RIGHT_EYE_FULL = [362, 263, 386, 374, 380, 381, 382, 362]
    LEFT_IRIS = [468, 469, 470, 471]
    RIGHT_IRIS = [473, 474, 475, 476]

    smooth_len = 5
    horiz_buffer = deque(maxlen=smooth_len)
    vert_buffer = deque(maxlen=smooth_len)
    trust_buffer = deque(maxlen=smooth_len)
    ear_buffer = deque(maxlen=10)

    CONSEC_FRAMES = 3
    blink_count = 0
    blink_frame = 0
    blink_state = False
    ear_baseline = None
    start_time = time.time()
    all_trust_scores = []

    def eye_aspect_ratio(landmarks, eye_indices, w, h):
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
        v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        h1 = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (v1 + v2) / (2.0 * h1)

    def get_iris_center(landmarks, iris_indices, w, h):
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in iris_indices]
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        return cx, cy

    def get_gaze_ratio(landmarks, eye_indices, iris_indices, w, h):
        eye_x = [landmarks[i].x * w for i in eye_indices]
        eye_y = [landmarks[i].y * h for i in eye_indices]
        eye_left, eye_right = min(eye_x), max(eye_x)
        eye_top, eye_bottom = min(eye_y), max(eye_y)
        cx, cy = get_iris_center(landmarks, iris_indices, w, h)
        horiz_ratio = (cx - eye_left) / (eye_right - eye_left + 1e-6)
        vert_ratio = 1 - ((cy - eye_top) / (eye_bottom - eye_top + 1e-6))
        return horiz_ratio, vert_ratio

    blink = ctrl.Antecedent(np.arange(0, 61, 1), 'blink_rate')
    gaze = ctrl.Antecedent(np.arange(0, 101, 1), 'gaze')
    emotion = ctrl.Antecedent(np.arange(1, 3, 1), 'emotion')
    trust = ctrl.Consequent(np.arange(0, 101, 1), 'trust')

    blink['low'] = fuzz.trimf(blink.universe, [0, 0, 20])
    blink['medium'] = fuzz.trimf(blink.universe, [10, 30, 50])
    blink['high'] = fuzz.trimf(blink.universe, [30, 60, 60])
    gaze['center'] = fuzz.trimf(gaze.universe, [40, 50, 60])
    gaze['away'] = fuzz.trimf(gaze.universe, [0, 40, 50])
    gaze['away_right'] = fuzz.trimf(gaze.universe, [50, 60, 100])
    emotion['low'] = fuzz.trimf(emotion.universe, [1, 1, 1.5])
    emotion['high'] = fuzz.trimf(emotion.universe, [1.5, 2, 2])
    trust['low'] = fuzz.trimf(trust.universe, [0, 0, 50])
    trust['medium'] = fuzz.trimf(trust.universe, [25, 50, 75])
    trust['high'] = fuzz.trimf(trust.universe, [50, 100, 100])

    rule1 = ctrl.Rule(blink['low'] & gaze['center'] & emotion['high'], trust['high'])
    rule2 = ctrl.Rule(blink['medium'] & gaze['center'] & emotion['high'], trust['medium'])
    rule3 = ctrl.Rule(blink['high'] | gaze['away'] | emotion['low'], trust['low'])
    trust_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

    cap = cv2.VideoCapture(0)
    print("Press 'q' to stop the analysis...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        emotion_val = 2
        horiz_ratio, vert_ratio = 0.5, 0.5
        label = "neutral"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_EAR = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
                right_EAR = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
                avg_EAR = (left_EAR + right_EAR) / 2.0
                ear_buffer.append(avg_EAR)
                smooth_EAR = np.mean(ear_buffer)

                if 'ear_baseline' not in locals() or ear_baseline is None:
                    ear_baseline = smooth_EAR
                ear_baseline = 0.98 * ear_baseline + 0.02 * smooth_EAR
                dynamic_thresh = ear_baseline * 0.75

                if smooth_EAR < dynamic_thresh:
                    blink_frame += 1
                    blink_state = True
                else:
                    if blink_state and blink_frame >= CONSEC_FRAMES:
                        blink_count += 1
                    blink_frame = 0
                    blink_state = False

                l_h, l_v = get_gaze_ratio(face_landmarks.landmark, LEFT_EYE_FULL, LEFT_IRIS, w, h)
                r_h, r_v = get_gaze_ratio(face_landmarks.landmark, RIGHT_EYE_FULL, RIGHT_IRIS, w, h)
                horiz_ratio = (l_h + r_h) / 2
                horiz_buffer.append(horiz_ratio)
                smooth_h = np.mean(horiz_buffer)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                x_min = int(min([lm.x for lm in face_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in face_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in face_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in face_landmarks.landmark]) * h)
                roi = gray[y_min:y_max, x_min:x_max]
                if roi.size != 0:
                    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                    roi = roi.astype('float32') / 255.0
                    roi = np.expand_dims(roi, (0, -1))
                    preds = model.predict(roi, verbose=0)
                    label = CLASSES[np.argmax(preds)]
                    emotion_val = 2 if label in ['happy', 'neutral'] else 1

        elapsed_time = time.time() - start_time
        minutes = elapsed_time / 60.0
        blink_rate = blink_count / minutes if minutes > 0 else 0
        blink_rate = np.clip(blink_rate, 0, 60)

        trust_sim = ctrl.ControlSystemSimulation(trust_ctrl)
        trust_sim.input['blink_rate'] = blink_rate
        trust_sim.input['gaze'] = smooth_h * 100
        trust_sim.input['emotion'] = emotion_val
        try:
            trust_sim.compute()
            trust_score = trust_sim.output['trust']
        except:
            trust_score = 50

        trust_buffer.append(trust_score)
        smooth_trust = np.mean(trust_buffer)
        all_trust_scores.append(smooth_trust)

        cv2.putText(frame, f"Emotion: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Trust: {smooth_trust:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks/min: {blink_rate:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gaze: {'Left' if smooth_h < 0.45 else 'Right' if smooth_h > 0.55 else 'Center'}",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Facial Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(all_trust_scores) > 0:
        final_trust = np.mean(all_trust_scores)
        print(f"\nFinal Trust Score: {final_trust:.2f}")
    else:
        print("\nNo face detected long enough to compute trust score.")
    record_metrics(all_trust_scores, start_time)

    return "Microexpression Analysis Completed ✅"



if __name__ == "__main__":
    run_analysis()
