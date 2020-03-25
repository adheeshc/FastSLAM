# -*- coding: utf-8 -*-
#   ___      _ _                    _     
#  / _ \    | | |                  | |    
# / /_\ \ __| | |__   ___  ___  ___| |__  
# |  _  |/ _` | '_ \ / _ \/ _ \/ __| '_ \ 
# | | | | (_| | | | |  __/  __/\__ \ | | |
# \_| |_/\__,_|_| |_|\___|\___||___/_| |_|
# Date:   2020-03-25 02:04:16
# Last Modified time: 2020-03-25 02:57:54

from fast_slam import FastSLAM
from aux import Constants,Particle,Motion_model,Landmark

if __name__=="__main__":

	print(__file__ + " start!!")

	cons=Constants()
	mm=Motion_model()
	lm=Landmark()

	fsl=FastSLAM(cons,mm,lm)
	fsl.plot(True)