# -*- coding: utf-8 -*-
#   ___      _ _                    _     
#  / _ \    | | |                  | |    
# / /_\ \ __| | |__   ___  ___  ___| |__  
# |  _  |/ _` | '_ \ / _ \/ _ \/ __| '_ \ 
# | | | | (_| | | | |  __/  __/\__ \ | | |
# \_| |_/\__,_|_| |_|\___|\___||___/_| |_|
# Date:   2020-03-24 19:50:39
# Last Modified time: 2020-03-25 03:03:33

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import random

class Constants():
	def __init__(self):
		self.cov_params()
		self.sim_params()
		lm=Landmark()
		self.n_landmarks=lm.len()[0]
		

	def cov_params(self):
		self.Q=np.diag([3,np.deg2rad(10)])**2		#process cov
		self.R=np.diag([1,np.deg2rad(20)])**2		#measurement cov

		self.Q_sim=np.diag([0.3,np.deg2rad(2)])**2	#process cov
		self.R_sim=np.diag([0.5,np.deg2rad(10)])**2	#measurement cov

		self.theta_offset=0.01						#noise
		self.dt=0.1									#time 

	def sim_params(self):	
		self.sim_time=50						#sim time
		self.max_range=20						#max observation range
		self.state_size=3						#state size (x,y,theta)
		self.landmark_size=2					#landmark state size (x,y)
		self.n_particles=100					#number of particles
		self.n_resample=self.n_particles/1.5	#number of particles for resampling

class Particle():
	def __init__(self,n_landmark):
		cons=Constants()
		self.w=1/cons.n_particles
		self.x=0
		self.y=0
		self.theta=0
		self.lm=np.zeros((n_landmark,cons.landmark_size))								#landmark positions (x,y)
		self.lm_cov=np.zeros((n_landmark*cons.landmark_size,cons.landmark_size))		#landmark covariance

class Landmark():
	def __init__(self):
		self.landmark=np.array([[10,-2],
							[15,10],
							[-10,0],
							[12,15],
							[5,15],
							[-5,20],
							[-5,5],
							[-10,15]
							])

	def len(self):
		return np.shape(self.landmark)

class Motion_model():
	def __init__(self):
		self.cons=Constants()

	def model(self,x,u):
		self.F=np.eye(3)

		self.B=np.zeros((3,2))
		self.B[0][0]=self.cons.dt*math.cos(x[2,0])
		self.B[1][0]=self.cons.dt*math.sin(x[2,0])
		self.B[2][1]=self.cons.dt
		
		x=self.F@x+self.B@u
		x[2,0]=self.convert_angle(x[2,0])

		return x

	def convert_angle(self,angle):
		return (angle+math.pi)%(2*math.pi)-math.pi

