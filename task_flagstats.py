import numpy as np
import matplotlib.pyplot as plt
from __casac__ import *
import os
import acquire 
import diagnostics
#smoothwindow=90,threshold=1,navg=12
def flagstats(vis='',outname='',scan='',excludecsqrbl=False,navgtime=12,smoothwindow=90,narrowbandthreshold=1,reprocess=False):
	np.seterr(divide='ignore', invalid='ignore')
	np.warnings.filterwarnings('ignore')
	narrowbandthreshold/=100.0
	if(not reprocess):
		acquire.getstat(vis=vis,outname=outname,scan=scan)
	diagnostics.getdiagnostics(statfolder=outname,excludecsqrbl=excludecsqrbl,smoothwindow=smoothwindow,threshold=narrowbandthreshold,navg=navgtime)
	
