# -*- coding: utf-8 -*-
#   ___      _ _                    _     
#  / _ \    | | |                  | |    
# / /_\ \ __| | |__   ___  ___  ___| |__  
# |  _  |/ _` | '_ \ / _ \/ _ \/ __| '_ \ 
# | | | | (_| | | | |  __/  __/\__ \ | | |
# \_| |_/\__,_|_| |_|\___|\___||___/_| |_|
# Date:   2020-03-24 19:48:21
# Last Modified time: 2020-03-25 03:13:38

import numpy as np
import matplotlib.pyplot as plt
import math
from aux import Constants,Particle,Motion_model,Landmark

class FastSLAM():
	def __init__(self,cons,mm,lm):
		self.cons=cons
		self.mm=mm
		self.lm=lm

	def calc_input(self,time):
		if time<=3.0:
			v=0.0
			w=0.0
		else:
			v=1.0
			w=0.1

		u = np.array([v, w]).reshape(2, 1)

		return u

	def observation(self,x_true, xd, u, landmark):
		#FIND TRUE VALUE
		x_true = self.mm.model(x_true, u)

		#ADD NOISE TO OBSERVATION
		z = np.zeros((3, 0))
		for i in range(len(landmark[:, 0])):
			dx=landmark[i, 0]-x_true[0, 0]
			dy=landmark[i, 1]-x_true[1, 0]
			d=math.sqrt(dx**2+dy**2)
			angle=self.convert_angle(math.atan2(dy,dx)-x_true[2, 0])
			if d<= self.cons.max_range:
				d_n=d+np.random.randn()*self.cons.Q_sim[0, 0]  			# add noise
				angle_n=angle+np.random.randn()*self.cons.Q_sim[1, 1]  	# add noise
				zi=np.array([d_n,self.convert_angle(angle_n),i]).reshape(3, 1)
				z=np.hstack((z,zi))
		
		#ADD NOISE TO INPUT
		ud1=u[0,0]+np.random.randn()*self.cons.R_sim[0, 0]
		ud2=u[1,0]+np.random.randn()*self.cons.R_sim[1, 1]+self.cons.theta_offset
		ud=np.array([ud1,ud2]).reshape(2,1)
		xd=self.mm.model(xd,ud)

		return x_true, z, xd, ud

	def algorithm(self,particles,u,z):
		particles=self.predict(particles,u)
		particles=self.update(particles,z)
		particles=self.resample(particles)
		
		return particles
	
	def predict(self,particles, u):
		for i in range(self.cons.n_particles):
			px=np.zeros((self.cons.state_size, 1))
			px[0,0]=particles[i].x
			px[1,0]=particles[i].y
			px[2,0]=particles[i].theta

			ud=u+(np.random.randn(1,2)@self.cons.R).T  # add noise
			px=self.mm.model(px,ud)
			particles[i].x=px[0,0]
			particles[i].y=px[1,0]
			particles[i].theta=px[2,0]

		return particles


	def update(self,particles, z):
		for iz in range(len(z[0, :])):
			lmid = int(z[2,iz])
			for ip in range(self.cons.n_particles):
				#NEW LANDMARK FOUND
				if abs(particles[ip].lm[lmid,0])<= 0.01:
					particles[ip]=self.add_new_landmark(particles[ip],z[:,iz],self.cons.Q)
				#ALREADY KNOWN LANDMARK
				else:
					w=self.compute_weights(particles[ip],z[:, iz],self.cons.Q)
					particles[ip].w*=w
					particles[ip]=self.update_landmark(particles[ip],z[:, iz],self.cons.Q)
		
		return particles

	def convert_angle(self,angle):
		return (angle+math.pi)%(2*math.pi)-math.pi

	def add_new_landmark(self,particle, z, Q):
		lm_id = int(z[2])
		particle.lm[lm_id,0]=particle.x+z[0]*math.cos(self.convert_angle(particle.theta+z[1]))
		particle.lm[lm_id,1]=particle.y+z[0]*math.sin(self.convert_angle(particle.theta+z[1]))

		#COVARIANCE MAT
		G=np.array([[math.cos(self.convert_angle(particle.theta + z[1])), -z[0] * math.sin(self.convert_angle(particle.theta + z[1]))],
					[math.sin(self.convert_angle(particle.theta + z[1])), z[0] * math.cos(self.convert_angle(particle.theta + z[1]))]])
		particle.lm_cov[2*lm_id:2*lm_id+2]=G@Q@G.T

		return particle


	def compute_weights(self,particle, z, Q):
		lm_id=int(z[2])
		xf=np.array(particle.lm[lm_id,:]).reshape(2,1)
		Pf=np.array(particle.lm_cov[2*lm_id:2*lm_id+2])
		zp,hv,hf,sf=self.compute_jacobians(particle, xf, Pf, Q)

		dx=z[0:2].reshape(2,1)-zp
		dx[1,0]=self.convert_angle(dx[1,0])

		try:
			s_inv=np.linalg.inv(sf)
		except np.linalg.linalg.LinAlgError:
			print("singuler")
			return 1

		num=math.exp(-0.5*dx.T@s_inv@dx)
		den=2*math.pi*math.sqrt(np.linalg.det(sf))
		w = num / den

		return w

	def normalize_weights(self,particles):
		sum_w = sum([p.w for p in particles])
		try:
			for i in range(self.cons.n_particles):
				particles[i].w /= sum_w
		except ZeroDivisionError:
			for i in range(self.cons.n_particles):
				particles[i].w = 1.0 / self.cons.n_particles
			return particles

		return particles

	def compute_jacobians(self,particle, xf, Pf, Q):
		dx=xf[0, 0]-particle.x
		dy=xf[1, 0]-particle.y
		d2=dx**2+dy**2
		d=math.sqrt(d2)

		zp=np.array([d,self.convert_angle(math.atan2(dy,dx)-particle.theta)]).reshape(2, 1)
		hv=np.array([[-dx/d,-dy/d,0],[dy/d2,-dx/d2,-1]])
		hf=np.array([[dx/d,dy/d],[-dy/d2,dx/d2]])
		sf=hf@Pf@hf.T+Q

		return zp, hv, hf, sf


	def update_landmark(self,particle, z, Q):
		lm_id=int(z[2])
		xf=np.array(particle.lm[lm_id, :]).reshape(2, 1)
		Pf=np.array(particle.lm_cov[2*lm_id:2*lm_id+2,:])
		zp,hv,hf,sf=self.compute_jacobians(particle, xf, Pf, Q)
		dz=z[0:2].reshape(2, 1)-zp
		dz[1, 0]=self.convert_angle(dz[1, 0])
		xf,Pf=self.update_with_cholesky(xf,Pf,dz,Q,hf)
		particle.lm[lm_id,:]=xf.T
		particle.lm_cov[2*lm_id:2*lm_id+2,:]=Pf

		return particle

	def update_with_cholesky(self,xf,Pf,v,Q,hf):
		PHt=Pf@hf.T
		S=hf@PHt+Q
		S=(S+S.T)*0.5
		S_chol=np.linalg.cholesky(S).T
		S_cinv=np.linalg.inv(S_chol)

		W1=PHt@S_cinv
		W=W1@S_cinv.T

		x=xf+W@v
		P=Pf-W1@W1.T

		return x, P

	def resample(self,particles):
		particles = self.normalize_weights(particles)
		pw = []
		for i in range(self.cons.n_particles):
			pw.append(particles[i].w)
		pw = np.array(pw)
		n_effective = 1.0 / (pw@pw.T)

		if n_effective < self.cons.n_resample:
			w_cum=np.cumsum(pw)
			base=np.cumsum(pw*0.0+1/self.cons.n_particles)-1/self.cons.n_particles
			resampleid=base+np.random.rand(base.shape[0])/self.cons.n_particles

			inds = []
			ind = 0
			for i in range(self.cons.n_particles):
				while ((ind < w_cum.shape[0] - 1) and (resampleid[i] > w_cum[ind])):
					ind += 1
				inds.append(ind)
			temp_particles = particles[:]

			for i in range(len(inds)):
				particles[i].x=temp_particles[inds[i]].x
				particles[i].y=temp_particles[inds[i]].y
				particles[i].theta=temp_particles[inds[i]].theta
				particles[i].lm=temp_particles[inds[i]].lm[:,:]
				particles[i].lm_cov=temp_particles[inds[i]].lm_cov[:,:]
				particles[i].w=1/self.cons.n_particles

		return particles

	def calc_final_state(self,particles):
		x_est=np.zeros((self.cons.state_size,1))
		particles=self.normalize_weights(particles)
		for i in range(self.cons.n_particles):
			x_est[0,0]+=particles[i].w*particles[i].x
			x_est[1,0]+=particles[i].w*particles[i].y
			x_est[2,0]+=particles[i].w*particles[i].theta
		x_est[2,0]=self.convert_angle(x_est[2,0])

		return x_est

	def plot(self,record=False):
		t = 0
		
		# STATE VECTOR 
		x_est=np.zeros((self.cons.state_size,1))  		# SLAM estimation
		x_true=np.zeros((self.cons.state_size,1))  		# True state
		x_DR=np.zeros((self.cons.state_size,1))  		# Dead reckoning
		
		# TRACK VECTORS
		h_est=x_est
		h_true=x_true
		h_DR=x_true

		particles=[Particle(len(self.lm.landmark)) for i in range(self.cons.n_particles)]

		
		while self.cons.sim_time>=t:
			t+=self.cons.dt
			u=self.calc_input(t)

			x_true,z,x_DR,ud=self.observation(x_true, x_DR, u, self.lm.landmark)
			final=self.algorithm(particles,ud,z)
			particles=final

			x_est=self.calc_final_state(particles)
			x_state=x_est[0: self.cons.state_size]

			h_est=np.hstack((h_est, x_state))
			h_DR=np.hstack((h_DR, x_DR))
			h_true=np.hstack((h_true, x_true))
			
			#PLOT
			plt.cla()
			plt.plot(self.lm.landmark[:,0],self.lm.landmark[:,1],"ok",zorder=1,ms=20)
			plt.plot(self.lm.landmark[:,0],self.lm.landmark[:,1],"xr",zorder=3,label="True Centre")
			for i in range(self.cons.n_particles):
				plt.plot(particles[i].lm[:, 0],particles[i].lm[:,1],"xy",ms=5,zorder=2)
			plt.plot(particles[0].lm[:, 0],particles[0].lm[:,1],"xy",ms=5,zorder=2,label="Estimated Centre")

			plt.plot(h_true[0,:],h_true[1, :],"-b",zorder=6,lw=4,label="True Path")
			plt.plot(h_DR[0,:],h_DR[1, :],"-k",zorder=5,label="Dead Reckoning")
			plt.plot(h_est[0,:],h_est[1, :],"-r",zorder=7,label="SLAM Estimation")
			plt.plot(x_est[0],x_est[1],"xk",lw=5,zorder=4)
			plt.axis([np.amin(self.lm.landmark,axis=0)[0]-5,np.amax(self.lm.landmark,axis=0)[0]+10,np.amin(self.lm.landmark,axis=0)[1]-5,np.amax(self.lm.landmark,axis=0)[1]+10])
			plt.grid(True)
			plt.title("FastSLAM - Particle Filter SLAM")
			plt.legend(loc=1)
			if record:
				plt.pause(5)
				record=False
			
			plt.pause(0.001)
