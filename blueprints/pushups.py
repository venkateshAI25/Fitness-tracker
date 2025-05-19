from flask import Blueprint, Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Initialize Flask app
pushups_bp = Blueprint('pushups', __name__, template_folder='../templates')

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Vector formation
    ba = a - b
    bc = c - b
    
    # Cosine law for angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    # Convert to degrees
    angle = np.degrees(angle)
    
    return angle

# Initialize variables for pushups
pushup_counter = 0
pushup_stage = None
feedback_text = "Position yourself in frame"
last_feedback_time = time.time()  # Initialize with current time
feedback_duration = 2  # seconds to display each feedback
confidence_threshold = 0.5  # Minimum confidence for detection

# Improved angle smoothing using deque for faster operations
smoothing_window_size = 10
elbow_angle_buffer = deque(maxlen=smoothing_window_size)
shoulder_angle_buffer = deque(maxlen=smoothing_window_size)
hip_angle_buffer = deque(maxlen=smoothing_window_size)

# Form check parameters
min_elbow_angle = 70  # At bottom of pushup
max_elbow_angle = 160  # At top of pushup

# ===== IMPROVED REP COUNTING PARAMETERS =====
# More robust thresholds for rep detection
TOP_POSITION_THRESHOLD = 150  # Angle threshold for top position (more lenient)
BOTTOM_POSITION_THRESHOLD = 90  # Angle threshold for bottom position

# Stability requirements
STABILITY_FRAMES_TOP = 5  # Frames needed at top position
STABILITY_FRAMES_BOTTOM = 3  # Frames needed at bottom position (less strict)

# Movement tracking
MIN_ANGLE_CHANGE = 3  # Minimum angle change to consider movement
MIN_REP_DEPTH = 50  # Minimum elbow angle change to consider a valid rep
MIN_REP_INTERVAL = 0.5  # Minimum time between reps (seconds)

# State tracking
pushup_states = {
    'IDLE': 0,
    'TOP_POSITION': 1,
    'GOING_DOWN': 2,
    'BOTTOM_POSITION': 3,
    'GOING_UP': 4
}
current_state = pushup_states['IDLE']
state_entry_time = time.time()
frames_in_current_state = 0
last_rep_time = 0
max_angle_in_rep = 0
min_angle_in_rep = 180

# Angle history for trend analysis
angle_history = deque(maxlen=30)
movement_direction = "none"  # "up", "down", or "none"
# ================================================

def live_video_feed():
    global pushup_counter, pushup_stage, feedback_text, last_feedback_time
    global elbow_angle_buffer, shoulder_angle_buffer, hip_angle_buffer
    global current_state, state_entry_time, frames_in_current_state, last_rep_time
    global max_angle_in_rep, min_angle_in_rep, angle_history, movement_direction
    
    cap = cv2.VideoCapture(0)
    
    # Set higher resolution if camera supports it
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # Use a more accurate model
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get the current time for feedback timing
            current_time = time.time()
            
            # Resize for faster processing while maintaining aspect ratio
            frame_height, frame_width = frame.shape[:2]
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                if results.pose_landmarks is None:
                    raise Exception("No pose detected")
                
                landmarks = results.pose_landmarks.landmark
                
                # Get specific landmarks for pushup tracking
                # We'll track shoulders, elbows, wrists, and hips
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                
                # Check if enough landmarks are visible for pushup detection
                enough_visibility = (
                    left_shoulder.visibility > confidence_threshold and
                    left_elbow.visibility > confidence_threshold and
                    left_wrist.visibility > confidence_threshold and
                    left_hip.visibility > confidence_threshold and
                    right_shoulder.visibility > confidence_threshold and
                    right_elbow.visibility > confidence_threshold and
                    right_wrist.visibility > confidence_threshold and
                    right_hip.visibility > confidence_threshold
                )
                
                if not enough_visibility:
                    # Reset rep tracking if visibility is lost
                    current_state = pushup_states['IDLE']
                    frames_in_current_state = 0
                    
                    if current_time - last_feedback_time > feedback_duration:
                        feedback_text = "Position your body so we can see you fully"
                        last_feedback_time = current_time
                    raise Exception("Not enough landmarks visible")
                
                # Convert normalized coordinates to pixel values for visualization
                def get_pixel_coords(landmark):
                    return (int(landmark.x * frame_width), int(landmark.y * frame_height))
                
                left_shoulder_px = get_pixel_coords(left_shoulder)
                left_elbow_px = get_pixel_coords(left_elbow)
                left_wrist_px = get_pixel_coords(left_wrist)
                left_hip_px = get_pixel_coords(left_hip)
                left_knee_px = get_pixel_coords(left_knee)
                
                right_shoulder_px = get_pixel_coords(right_shoulder)
                right_elbow_px = get_pixel_coords(right_elbow)
                right_wrist_px = get_pixel_coords(right_wrist)
                right_hip_px = get_pixel_coords(right_hip)
                right_knee_px = get_pixel_coords(right_knee)
                
                # Calculate angles for pushup form checking
                # We'll use the average of left and right sides for more stability
                left_elbow_angle = calculate_angle(
                    [left_shoulder.x, left_shoulder.y],
                    [left_elbow.x, left_elbow.y],
                    [left_wrist.x, left_wrist.y]
                )
                
                right_elbow_angle = calculate_angle(
                    [right_shoulder.x, right_shoulder.y],
                    [right_elbow.x, right_elbow.y],
                    [right_wrist.x, right_wrist.y]
                )
                
                # Use the better visible side for primary angle calculation
                if left_elbow.visibility > right_elbow.visibility:
                    primary_elbow_angle = left_elbow_angle
                else:
                    primary_elbow_angle = right_elbow_angle
                
                # Calculate shoulder to hip angle (torso alignment)
                left_shoulder_angle = calculate_angle(
                    [left_elbow.x, left_elbow.y],
                    [left_shoulder.x, left_shoulder.y],
                    [left_hip.x, left_hip.y]
                )
                
                right_shoulder_angle = calculate_angle(
                    [right_elbow.x, right_elbow.y],
                    [right_shoulder.x, right_shoulder.y],
                    [right_hip.x, right_hip.y]
                )
                
                avg_shoulder_angle = (left_shoulder_angle + right_shoulder_angle) / 2
                
                # Calculate hip to knee angle (for back alignment)
                left_hip_angle = calculate_angle(
                    [left_shoulder.x, left_shoulder.y],
                    [left_hip.x, left_hip.y],
                    [left_knee.x, left_knee.y]
                )
                
                right_hip_angle = calculate_angle(
                    [right_shoulder.x, right_shoulder.y],
                    [right_hip.x, right_hip.y],
                    [right_knee.x, right_knee.y]
                )
                
                avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
                
                # Add to smoothing buffers
                elbow_angle_buffer.append(primary_elbow_angle)
                shoulder_angle_buffer.append(avg_shoulder_angle)
                hip_angle_buffer.append(avg_hip_angle)
                
                # Calculate smoothed angles
                smoothed_elbow_angle = sum(elbow_angle_buffer) / len(elbow_angle_buffer)
                smoothed_shoulder_angle = sum(shoulder_angle_buffer) / len(shoulder_angle_buffer)
                smoothed_hip_angle = sum(hip_angle_buffer) / len(hip_angle_buffer)
                
                # Add to angle history for trend analysis
                angle_history.append(smoothed_elbow_angle)
                
                # Determine movement direction
                if len(angle_history) >= 5:
                    recent_angles = list(angle_history)[-5:]
                    if recent_angles[0] - recent_angles[-1] > MIN_ANGLE_CHANGE:
                        movement_direction = "down"
                    elif recent_angles[-1] - recent_angles[0] > MIN_ANGLE_CHANGE:
                        movement_direction = "up"
                    else:
                        movement_direction = "stable"
                
                # Draw key points and lines for visualization
                # Shoulders to elbows
                cv2.line(frame, left_shoulder_px, left_elbow_px, (255, 255, 0), 3)
                cv2.line(frame, right_shoulder_px, right_elbow_px, (255, 255, 0), 3)
                
                # Elbows to wrists
                cv2.line(frame, left_elbow_px, left_wrist_px, (255, 255, 0), 3)
                cv2.line(frame, right_elbow_px, right_wrist_px, (255, 255, 0), 3)
                
                # Shoulders to hips
                cv2.line(frame, left_shoulder_px, left_hip_px, (255, 255, 0), 3)
                cv2.line(frame, right_shoulder_px, right_hip_px, (255, 255, 0), 3)
                
                # Hips to knees
                cv2.line(frame, left_hip_px, left_knee_px, (255, 255, 0), 3)
                cv2.line(frame, right_hip_px, right_knee_px, (255, 255, 0), 3)
                
                # Draw key points
                for point in [left_shoulder_px, left_elbow_px, left_wrist_px, left_hip_px, left_knee_px,
                             right_shoulder_px, right_elbow_px, right_wrist_px, right_hip_px, right_knee_px]:
                    cv2.circle(frame, point, 8, (0, 0, 255), -1)
                
                # Display angles
                cv2.putText(frame, f"Elbow: {smoothed_elbow_angle:.1f}", 
                            (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Back: {smoothed_hip_angle:.1f}", 
                            (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # ===== IMPROVED PUSHUP STATE MACHINE =====
                
                # Display current state, movement, and frames in state
                state_names = {v: k for k, v in pushup_states.items()}
                cv2.putText(frame, f"State: {state_names[current_state]}", 
                            (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Movement: {movement_direction}", 
                            (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Frames: {frames_in_current_state}", 
                            (10, 180), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # STATE MACHINE LOGIC
                # State 0: IDLE - Waiting for the person to get in position
                if current_state == pushup_states['IDLE']:
                    if smoothed_elbow_angle > TOP_POSITION_THRESHOLD:
                        current_state = pushup_states['TOP_POSITION']
                        frames_in_current_state = 1
                        state_entry_time = current_time
                        max_angle_in_rep = smoothed_elbow_angle
                        feedback_text = "Good starting position"
                        last_feedback_time = current_time
                    else:
                        frames_in_current_state = 0
                
                # State 1: TOP_POSITION - Person is at the top of the pushup
                elif current_state == pushup_states['TOP_POSITION']:
                    frames_in_current_state += 1
                    
                    # Update max angle seen in this rep
                    max_angle_in_rep = max(max_angle_in_rep, smoothed_elbow_angle)
                    
                    # If person starts going down
                    if movement_direction == "down" and smoothed_elbow_angle < TOP_POSITION_THRESHOLD:
                        current_state = pushup_states['GOING_DOWN']
                        frames_in_current_state = 1
                        feedback_text = "Going down, good form"
                        last_feedback_time = current_time
                    
                    # If we've been stable at the top for a while, provide feedback
                    elif frames_in_current_state > 30 and current_time - last_feedback_time > feedback_duration:
                        feedback_text = "Lower yourself to start a pushup"
                        last_feedback_time = current_time
                
                # State 2: GOING_DOWN - Person is lowering into the pushup
                elif current_state == pushup_states['GOING_DOWN']:
                    frames_in_current_state += 1
                    
                    # If person reaches bottom position
                    if smoothed_elbow_angle < BOTTOM_POSITION_THRESHOLD:
                        current_state = pushup_states['BOTTOM_POSITION']
                        frames_in_current_state = 1
                        min_angle_in_rep = smoothed_elbow_angle
                        feedback_text = "Good depth"
                        last_feedback_time = current_time
                    
                    # If person starts going back up without reaching bottom
                    elif movement_direction == "up" and frames_in_current_state > 5:
                        current_state = pushup_states['GOING_UP']
                        frames_in_current_state = 1
                        min_angle_in_rep = min(min_angle_in_rep, smoothed_elbow_angle)
                        feedback_text = "Try to go deeper next time"
                        last_feedback_time = current_time
                    
                    # Going down too long - possible problem
                    elif frames_in_current_state > 20 and current_time - last_feedback_time > feedback_duration:
                        feedback_text = "Lower to complete the rep"
                        last_feedback_time = current_time
                
                # State 3: BOTTOM_POSITION - Person is at the bottom of the pushup
                elif current_state == pushup_states['BOTTOM_POSITION']:
                    frames_in_current_state += 1
                    
                    # Update min angle seen in this rep
                    min_angle_in_rep = min(min_angle_in_rep, smoothed_elbow_angle)
                    
                    # If person starts going up
                    if movement_direction == "up" and smoothed_elbow_angle > BOTTOM_POSITION_THRESHOLD:
                        current_state = pushup_states['GOING_UP']
                        frames_in_current_state = 1
                        feedback_text = "Push up, you got this!"
                        last_feedback_time = current_time
                    
                    # If we've been at the bottom too long, provide feedback
                    elif frames_in_current_state > 15 and current_time - last_feedback_time > feedback_duration:
                        feedback_text = "Push back up"
                        last_feedback_time = current_time
                
                # State 4: GOING_UP - Person is pushing up from the pushup
                elif current_state == pushup_states['GOING_UP']:
                    frames_in_current_state += 1
                    
                    # If person reaches top position again - complete rep
                    if smoothed_elbow_angle > TOP_POSITION_THRESHOLD:
                        # Check if this was a valid rep (enough depth)
                        rep_depth = max_angle_in_rep - min_angle_in_rep
                        
                        if rep_depth > MIN_REP_DEPTH and current_time - last_rep_time > MIN_REP_INTERVAL:
                            pushup_counter += 1
                            feedback_text = f"Rep {pushup_counter} complete! Good job"
                            last_feedback_time = current_time
                            last_rep_time = current_time
                        elif rep_depth <= MIN_REP_DEPTH:
                            feedback_text = "Not deep enough for a complete rep"
                            last_feedback_time = current_time
                        
                        # Reset to top position for next rep
                        current_state = pushup_states['TOP_POSITION']
                        frames_in_current_state = 1
                        max_angle_in_rep = smoothed_elbow_angle
                        min_angle_in_rep = 180  # Reset for next rep
                    
                    # If we're going up too long - possible problem
                    elif frames_in_current_state > 20 and current_time - last_feedback_time > feedback_duration:
                        feedback_text = "Push all the way up"
                        last_feedback_time = current_time
                
                # Update pushup stage for visualization
                if current_state == pushup_states['TOP_POSITION'] or current_state == pushup_states['IDLE']:
                    pushup_stage = "up"
                elif current_state == pushup_states['BOTTOM_POSITION']:
                    pushup_stage = "down"
                elif current_state == pushup_states['GOING_DOWN']:
                    pushup_stage = "going down"
                elif current_state == pushup_states['GOING_UP']:
                    pushup_stage = "going up"
                
                # Form feedback
                if current_state != pushup_states['IDLE'] and smoothed_hip_angle < 160 and current_time - last_feedback_time > feedback_duration:
                    feedback_text = "Keep your back straight, don't sag"
                    last_feedback_time = current_time

            except Exception as e:
                # Reset state if person moves out of frame
                current_state = pushup_states['IDLE']
                frames_in_current_state = 0
                
                if current_time - last_feedback_time > feedback_duration:
                    feedback_text = "Position yourself in frame"
                    last_feedback_time = current_time

            # Display rep stage
            cv2.putText(frame, f"Stage: {pushup_stage if pushup_stage else 'none'}", 
                        (frame_width-200, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 255, 0) if pushup_stage == "up" else (0, 0, 255) if pushup_stage == "down" else (255, 255, 255), 
                        2)
            
            # Draw feedback box at the bottom
            cv2.rectangle(frame, (0, frame_height-70), (frame_width, frame_height), (0, 0, 0), -1)
            cv2.putText(frame, f'Push-ups: {pushup_counter}', (10, frame_height-40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, feedback_text, (10, frame_height-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw pose landmarks if available
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

            # Encode the frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()
        cv2.destroyAllWindows()

# Function to stream feedback video with text at the bottom
def feedback_display():
    global pushup_counter, feedback_text
    video_path = r'C:\Users\venkateshwaran\OneDrive\Documents\final (march ) AI (SRH)\final (march ) AI (SRH) after git\Ai\static\videos\WhatsApp Video 2024-12-09 at 17.11.08_a91de452.mp4'  # Update with your pushup reference video path
    cap = cv2.VideoCapture(video_path)

    # Check if video file exists and is opened successfully
    if not cap.isOpened():
        # Create a blank frame with error message if video can't be loaded
        while True:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Video file not found or cannot be opened", 
                        (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add counter and feedback
            cv2.rectangle(frame, (0, 320), (640, 360), (0, 0, 0), -1)
            cv2.putText(frame, f'Push-ups: {pushup_counter}', (10, 345), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, feedback_text, (220, 345), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            feedback_frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + feedback_frame + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    else:
        # Set video frame rate (fps)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30fps if unable to get proper fps

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video once it finishes
                continue

            # Resize the video frame to a smaller size
            frame = cv2.resize(frame, (640, 360))

            # Add feedback text to the bottom of the frame
            cv2.rectangle(frame, (0, 320), (640, 360), (0, 0, 0), -1)  # Black background for text
            cv2.putText(frame, f'Push-ups: {pushup_counter}', (10, 345), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, feedback_text, (220, 345), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Encode the frame
            ret, buffer = cv2.imencode('.jpg', frame)
            feedback_frame = buffer.tobytes()

            # Yield the frame to Flask
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + feedback_frame + b'\r\n')

            # Wait to control the frame rate
            cv2.waitKey(int(1000 / fps))  # Wait based on the original frame rate

    cap.release()

# Routes
@pushups_bp.route('/')
def index():
    return render_template('pushups.html')

@pushups_bp.route('/video_feed')
def video_feed():
    return Response(live_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@pushups_bp.route('/feedback')
def feedback():
    return Response(feedback_display(), mimetype='multipart/x-mixed-replace; boundary=frame')