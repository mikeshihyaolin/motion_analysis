# coding=utf-8
"""
@Author: Mike Lin
@File: demo_hand.py
@Time: 2021-04-15 10:00
@Last_update: 2021-04-29 11:00
"""
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class handDemo(object):
	def __init__(self):
		self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
		self.fig.subplots_adjust(hspace=0.5)
		self.leftAmplitude = [0]*30
		self.rightAmplitude = [0]*30

	def smooth(self, y, box_pts):
		box = np.ones(box_pts)/box_pts
		y_smooth = np.convolve(y, box, mode='same')
		return y_smooth

	def getAmplitude(self, hand_landmarks):
		thumb = np.asarray([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
		index = np.asarray([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
		distance = np.sqrt(np.sum(np.square(thumb-index)))
		return distance

	def plotAmplitude(self):
		self.ax1.clear()
		self.ax2.clear()
		# left = smooth(np.array(leftAmplitude),2)
		# right = smooth(np.array(rightAmplitude),2)
		left = self.leftAmplitude
		right = self.rightAmplitude

		self.ax1.plot(left, 'b--')
		self.ax1.grid(True)
		self.ax1.set_ylim(-0.2, 0.6)
		self.ax1.set_ylabel('Amplitude (Left)')
		self.ax1.set_xticks([], minor=False)
		self.ax1.set_yticks([], minor=False)

		self.ax2.plot(right, 'r--')
		self.ax2.set_ylim(-0.2 , 0.6)
		self.ax2.set_ylabel('Amplitude (Right)')
		self.ax2.set_xticks([], minor=False)
		self.ax2.set_yticks([], minor=False)
		plt.pause(0.01)

	def run(self):
		''' get camera '''
		cap = cv2.VideoCapture(0)
		''' detect hands '''
		with mp_hands.Hands(
			static_image_mode=True,
			max_num_hands=2,
			min_detection_confidence=0.7) as hands:		
			while cap.isOpened():
				ret, frame = cap.read()	
				results = hands.process(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),1))
				annotated_image = cv2.flip(frame.copy(), 1)
				self.leftAmplitude.pop(0)
				self.rightAmplitude.pop(0)
				if results.multi_hand_landmarks:
					for hand_landmarks in results.multi_hand_landmarks:
						mp_drawing.draw_landmarks(
							annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
						if "Left" in results.multi_handedness[0].classification[0].label and "right" not in results.multi_handedness[0].classification[0].label:
							score = self.getAmplitude(hand_landmarks)
							self.leftAmplitude.append(score)
							self.rightAmplitude.append(0)
						if "Right" in results.multi_handedness[0].classification[0].label and "left" not in results.multi_handedness[0].classification[0].label:
							score = self.getAmplitude(hand_landmarks)
							self.rightAmplitude.append(score)
							self.leftAmplitude.append(0)
				else:
					self.leftAmplitude.append(0)
					self.rightAmplitude.append(0)
				cv2.imshow('PD Live Demo', cv2.flip(annotated_image,1))
				if cv2.waitKey(10) & 0xFF == ord('q'):
					break
				''' Plot Amplitude '''
				self.plotAmplitude()

		cap.release()
		cv2.destroyAllWindows()			

if __name__ == '__main__': 
	demo = handDemo()
	demo.run()