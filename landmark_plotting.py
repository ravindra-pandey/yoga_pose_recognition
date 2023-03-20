import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.5)
def plot_pose(img):
    frame = img
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        frame = cv2.putText(frame, "No Points Found", (0, 10),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 00, 0), 2)
        return frame
    landmarks=results.pose_landmarks.landmark
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    return frame,landmarks
