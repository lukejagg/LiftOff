# AI Packages
import random

import cv2
import math
from typing import Mapping
import numpy as np
import mediapipe as mp
from test_data import NATE, BAD_NATE, DATA_SIZE
from network import model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

from subprocess import call

import threading
import pyttsx3
engine = pyttsx3.init()

class thread(threading.Thread):
    def __init__(self, text):
        threading.Thread.__init__(self)
        self.text = text

        # helper function to execute the threads

    def run(self):
        call(["python3", "speak.py", self.said])
        #engine.say(self.text)
        #engine.runAndWait()

# Data
# pose { pose_landmarks, pose_world_landmarks, segmentation_mask }
# landmark { x, y, z, visibility }

debug_index = 0
landmarks = None
screen_landmarks = None

print(f'Data Size: {DATA_SIZE} loaded')

#engine.say("I will speak this text")
#engine.runAndWait()


# Util Methods
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_RGB_CHANNELS = 3
WHITE_COLOR = (224,224,224)
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int):
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def draw(
    image,
    landmark_list,
    connections,
    landmark_drawing_spec,
    connection_drawing_spec):

  if not landmark_list:
    return
  image_rows, image_cols, _ = image.shape
  idx_to_coordinates = {}
  for idx, landmark in enumerate(landmark_list):
    if ((landmark.HasField('visibility') and
         landmark.visibility < _VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < _PRESENCE_THRESHOLD)):
      continue
    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
    if landmark_px:
      idx_to_coordinates[idx] = landmark_px
  if connections:
    num_landmarks = len(landmark_list)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections:
      start_idx = connection[0]
      end_idx = connection[1]
      if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
        raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')
      if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
        drawing_spec = connection_drawing_spec[connection] if isinstance(
            connection_drawing_spec, Mapping) else connection_drawing_spec
        cv2.line(image, idx_to_coordinates[start_idx],
                 idx_to_coordinates[end_idx], drawing_spec.color,
                 drawing_spec.thickness)
  # Draws landmark points after finishing the connection lines, which is
  # aesthetically better.
  if landmark_drawing_spec:
    for idx, landmark_px in idx_to_coordinates.items():
      drawing_spec = landmark_drawing_spec[idx] if isinstance(
          landmark_drawing_spec, Mapping) else landmark_drawing_spec
      # White circle border
      circle_border_radius = max(drawing_spec.circle_radius + 1,
                                 int(drawing_spec.circle_radius * 1.2))
      cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
                 drawing_spec.thickness)
      # Fill color into the circle
      cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                 drawing_spec.color, drawing_spec.thickness)

# Methods
def pos(landmark):
    return (landmark.x, landmark.y, landmark.z)
def getPos(index):
    return landmarks and pos(landmarks[index]) or (0, 0, 0)
def div(landmark, divider):
    return (landmark[0] / divider, landmark[1] / divider, landmark[2] / divider)
def sub(landmark1, landmark2):
    return (landmark1[0] - landmark2[0], landmark1[1] - landmark2[1], landmark1[2] - landmark2[2])
def average(a, b):
    return div(getPos(a) + getPos(b), 2)

averages = [0,0,0,0,0,0]
rate = 0.2

ttsTexts = [None,
         "Your feet are too close. Try to move them further out to a little more than shoulder width." ,
         "Your feet are too far out. Try to move them a little closer. ",
         "Do not round your back. Keep your back straight. ",
         "Move your hands a little closer to the middle near shoulder width. ",
         "Keep your back straight at the top of a rep. "]


def doSpeak():
    text = []
    for index, val in enumerate(averages):
        if val > 0.5 and ttsTexts[index] is not None:
            text.append(ttsTexts[index])
    said = "Your form is good. Keep it up! "
    if len(text) > 0:
        said = random.choice(text)

    t = thread(said)
    t.run()

def predict(data, image):
    if len(data) != DATA_SIZE or (screen_landmarks is None):
        print("Needs data to predict...")
        return

    m = screen_landmarks[7]
    if m.visibility < 0.5:
        return

    """"
    shape = image.shape
    screen_pos = (int(l.x * shape[1]), int(l.y * shape[0]))
    screen_pos2 = (int(m.x * shape[1]), int(m.y * shape[0]))
    delta = (screen_pos[0] - screen_pos2[0], screen_pos[1] - screen_pos2[1])
    #screen_pos = (screen_pos[0] + delta[0] * 2 + 30, screen_pos[1] + delta[1] * 2) # place text in direction of head
    #screen_pos = (screen_pos[0] + delta[0] * 1 + 30 + 80, screen_pos[1] + delta[1] * 1)
    """
    shape = image.shape
    screen_pos = (int(m.x * shape[1] + 50), int(m.y * shape[0] + 15))

    data = np.array(data).reshape((-1,1,DATA_SIZE))
    predictions = model.predict(data)
    out = predictions[0]
    print(f'Prediction: {out[0]:.2f}, {out[1]:.2f}')

    for index, val in enumerate(out):
        averages[index] = averages[index] + (val - averages[index]) * rate

    nums = averages
    texts = [None, "Move feet more out (shoulder width)", "Move feet closer together (shoulder width)", "Keep back straight (don't round your back)", "Move hands closer together", "Keep your back straight at the top"]
    text = []
    for index, val in enumerate(nums):
        if val > 0.5 and texts[index] is not None:
            text.append(texts[index])

    for index, t in enumerate(text):
        cv2.putText(image, t, (10, (index+1) * 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (40, 40, 240), 1, cv2.LINE_AA)

    v = int(nums[0] * 255)
    #cv2.putText(image, f'{out[0]:.2f}, {out[1]:.2f}, {out[2]:.2f}, {out[3]:.2f}, {out[4]:.2f}', screen_pos, cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(image, f'{nums[0]*100:.0f}%', screen_pos,
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (28, v, 255 - v), 1, cv2.LINE_AA)

    # Debug
    length = len(screen_landmarks)
    #l = landmarks[debug_index % length]
    #cv2.putText(image, f'{l.x:.2f},{l.y:.2f},{l.z:.2f}', (screen_pos[0], screen_pos[1] + 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    #cv2.putText(image, f'{screen_landmarks[0].z:.2f}', (screen_pos[0], screen_pos[1] + 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

def processData(results):
    #screenLandmarks = results.pose_landmarks
    #landmarks = results.pose_world_landmarks

    if not pose:
        print("No pose detected.")
        return

    positions = []

    return

def calculateHeight(results):
    return

def height(pos):
    total = -pos[1] * 369.565 - 334.913
    f = int(total // 12)
    i = total % 12
    return f, i

def packData(list):
    data = []
    for point in list:
        data.append(point[0])
        data.append(point[1])
        data.append(point[2])
    return data

doPredict = False

# Tracking Workflow
cam = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.75,min_tracking_confidence=0.25) as pose:
    while cam.isOpened():
        success, image = cam.read()
        if not success:
            print("No camera input read...")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        landmarks = results.pose_world_landmarks and results.pose_world_landmarks.landmark or None
        screen_landmarks = results.pose_landmarks and results.pose_landmarks.landmark or None

        data = processData(results)

        nose = getPos(0)
        mouth = average(9, 10)
        chest = average(11, 12)
        waist = average(23, 24)
        leftKnee = getPos(25)
        leftAnkle = getPos(27)
        leftToe = getPos(29)
        rightKnee = getPos(26)
        rightAnkle = getPos(28)
        rightToe = getPos(30)
        leftElbow = getPos(13)
        leftWrist = getPos(15)
        leftHand = average(17, 19)
        rightElbow = getPos(14)
        rightWrist = getPos(16)
        rightHand = average(18, 20)
        data = packData([nose, mouth, chest, waist, leftKnee, leftAnkle, leftToe, rightKnee, rightAnkle, rightToe, leftElbow, leftWrist, leftHand, rightElbow, rightWrist, rightHand])

        feet = div(getPos(31) + getPos(32), 2)
        dif = sub(nose, feet)
        h = height(dif)
        #print(f'{h[0]}\'{h[1]}"')

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(), connection_drawing_spec=mp_drawing_styles.DrawingSpec())

        # Highlight points:
        if results.pose_landmarks:
            length = len(results.pose_landmarks.landmark)
            m = debug_index % length
            landmark = results.pose_landmarks.landmark[m]
            vertex = [(landmark)]
            draw(image, vertex, None, landmark_drawing_spec=mp_drawing_styles.DrawingSpec(color=mp_drawing.GREEN_COLOR), connection_drawing_spec=mp_drawing_styles.DrawingSpec())

        if doPredict:
            predict(data, image)

        #cv2.imshow('LiftOff', image)#cv2.flip(image, 1))
        cv2.imshow('Liftoff', image)


        input = cv2.waitKey(1) & 0xFF
        if input == ord('d'):
            debug_index += 1
        if input == ord('a'):
            debug_index -= 1
        if input == ord('s'):
            break
        # if not input == 255:
        #     print(input & 0xFF)
        if input == 91:
            print(str(data) + ",")
            doSpeak()
            #print(f'{h[0]}\'{h[1]}"')
        if input == 93:
            doPredict = not doPredict
        if input == ord('q'):
            break

cam.release()