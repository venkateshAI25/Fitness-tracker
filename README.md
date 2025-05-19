# AI Fitness Trainer App

This is an AI-powered fitness trainer web app that provides real-time workout guidance, posture correction, and personalized diet suggestions based on user input. The app analyzes user movements through video input and compares them to correct forms to help improve exercise efficiency and avoid injuries.

## Features

- Real-time video analysis of workouts
- Posture/form correction for exercises (push-ups, squats, bicep curls)
- AI-powered feedback with visuals
- Personalized diet suggestions based on user weight
- Simple web-based interface with live and reference video screens

## Technologies Used

- *Python* (for AI analysis and video processing)
- *OpenCV* (for capturing and analyzing video frames)
- *MediaPipe* (for pose estimation and form detection)
- *Flask* (Python web framework to serve the AI backend)
- *HTML/CSS/JavaScript* (for frontend interface)
- *Heroku* (for deployment)

##Folder structure
AI-Fitness-Trainer/
│
├── pushup/
│   ├── pushup_detection.py              # Pose detection and feedback for push-ups
│   ├── pushup_video.mp4                 # Pre-recorded push-up guidance video
│   
│
├── squat/
│   ├── squat_detection.py               # Pose detection and feedback for squats
│   ├── squat_video.mp4                  # Pre-recorded squat guidance video
│   
│
├── bicep_curl/
│   ├── bicep_curl_detection.py          # Pose detection and feedback for bicep curls
│   ├── bicep_curl_video.mp4             # Pre-recorded bicep curl guidance video
│  
│
├── diet/
│   ├── diet_plan.py                     # Diet plan based on weight and goal
│   └── templates/
│       └── diet.html                    # Diet plan UI page
│
├── templates/
│   ├── index.html                       # Homepage with exercise and diet options
│   ├── pushup.html                      # Push-up detection interface
│   ├── squat.html                       # Squat detection interface
│   ├── bicep_curl.html                  # Bicep curl detection interface
│
│
├── app.py                               # Main Flask app routing to all pages
├── requirements.txt                     # Python dependencies
└── README.md                            # Project overview

##Demo Screenshots
MAIN DASHBOARD 
![image](https://github.com/user-attachments/assets/7f799d33-a2b4-432a-b79b-e6df5f7d0d27)

DIET PLAN 
 ![image](https://github.com/user-attachments/assets/a68cfadd-6cf2-41d0-8bea-afdaa14ab470)

PUSH-UPS 
![image](https://github.com/user-attachments/assets/0094ad1c-9892-4101-a049-051a874e7a09)

BICEPS CURL
![image](https://github.com/user-attachments/assets/2cc6d8bc-a6c0-4026-a824-a7a8d5f15661)

SQUATS
![image](https://github.com/user-attachments/assets/b9dea967-5ca6-4da7-a33e-4ec8df1fa2b5)
 
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/venkateshAI25/Fitness-tracker.git
   cd Fitness-tracker
