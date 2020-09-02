import numpy as np
import sys
import os 
from __casac__ import *


nant=30

def attachms(vis,scan,tblock):
	msobj=ms.ms()
	msobj.open(vis,nomodify=True)
	msobj.msselect({'scan':scan})
	msobj.iterinit(interval=tblock,adddefaultsortcolumns=False)
	endflag=msobj.iterorigin()
	return msobj,endflag

def writechanfreq(vis,fname,spw=0):
	tbobj=table.table()
	tbobj.open(vis+'/SPECTRAL_WINDOW')
	freq=tbobj.getcol('CHAN_FREQ')
	tbobj.close()
	np.savetxt(fname,freq[:,spw]/1e6,fmt='%.6f')

def writetimescan(vis,scan,fname):
	msobj=ms.ms()
	msobj.open(vis,nomodify=True)
	msobj.msselect({'scan':scan})
	data=msobj.getdata(['TIME','SCAN_NUMBER','DATA_DESC_ID'],ifraxis=True) 
	msobj.close()
	scan=data['scan_number']
	time=data['time']
	spw=data['data_desc_id']
	if(len(np.unique(spw))>1):
		print("ERROR: Select one spectral window at a time")
		return -1
	uniqscan=np.unique(scan)
	irec=np.arange(len(scan))
	obs=np.zeros(len(scan))
	iobs=0
	for iscan in range(len(uniqscan)):
		indx=scan==uniqscan[iscan]
		nlast=irec[indx][-1]
		obs[indx]=iobs
		if(iscan==len(uniqscan)-1):
			break
		delt=time[nlast+1]-time[nlast]
		if(delt>3600.0):
			iobs+=1
	np.savetxt(fname,np.column_stack((time,scan,obs)),fmt='%.10e %d %d')
	return len(time),spw[0]

def getantindx():
	ant1=[]
	ant2=[]
	for iant1 in range(0,nant):
		for iant2 in range(iant1+1,nant):
			ant1.append(iant1)
			ant2.append(iant2)
	ant1=np.array(ant1)
	ant2=np.array(ant2)
	return ant1,ant2


def getstat(vis,outname,scan='',tblock=300):
	if(os.path.isdir(outname)):
		print("Directory %s already exists."%outname)
		print("Please change outname or delete existing directory")
		return
	else:
		os.system("mkdir %s"%outname)

	nrecords,spw=writetimescan(vis,scan,outname+'/scaninfo.dat')	
	if(nrecords<0):
		return 
	writechanfreq(vis,outname+'/freq.dat',spw)
	nrecordscur=0
	ms,endflag=attachms(vis,scan=scan,tblock=tblock)
	data=ms.getdata(['FLAG'],ifraxis=True) 
	chanstat=np.zeros(data['flag'].shape[1:3])
	timestat_bl=np.zeros((0,data['flag'].shape[2]))
	timestat_freq=np.zeros((0,data['flag'].shape[1]))
	timestat_antfreq=np.zeros((nant,data['flag'].shape[1],0))
	endflag=ms.iterorigin()
	ant1,ant2=getantindx()

	progressbar=[]
	for i in range(0,100,10):
		progressbar.append(['%d'%i,'.','.','.','.'])

	progressbar=np.array(progressbar).flatten()
	progressbar=np.append(progressbar,['100'])
	progressindx=1
	sys.stdout.write(progressbar[0])
	sys.stdout.flush()

	while(endflag):
		flag=ms.getdata(['FLAG'],ifraxis=True)['flag']	
		if(len(flag.shape)<4):	
			flag=np.expand_dims(flag,axis=3)
		chanstat+=np.sum(np.sum(flag,axis=0),axis=2)
		timestat_bl=np.append(timestat_bl,np.mean(np.mean(flag,axis=0),axis=0).T,axis=0)
		timestat_freq=np.append(timestat_freq,np.mean(np.mean(flag,axis=0),axis=1).T,axis=0)
		antfreq=np.zeros((nant,flag.shape[1],flag.shape[3]))
		for iant in range(0,nant):
			mask=np.logical_or(ant1==iant,ant2==iant)
			antfreq[iant,:,:]=np.mean(np.mean(flag,axis=0)[:,mask,:],axis=1)
		timestat_antfreq=np.append(timestat_antfreq,antfreq,axis=2)
		endflag=ms.iternext()	
		nrecordscur+=flag.shape[3]
		curprogress=np.round((nrecordscur/float(nrecords))*50).astype("int")
		for i in range(progressindx,curprogress):			
			sys.stdout.write(progressbar[i])
			sys.stdout.flush()
		progressindx=curprogress
	sys.stdout.write(progressbar[-1])
	sys.stdout.flush()
	print('')

	ms.iterend()
	ms.close()
	chanstat/=float(2*timestat_bl.shape[0])
	np.savetxt(outname+'/chanstat.dat',chanstat,fmt='%.3f')
	np.savetxt(outname+'/timestat_bl.dat',timestat_bl,fmt='%.3f')
	np.savetxt(outname+'/timestat_freq.dat',timestat_freq,fmt='%.3f')
	timestat_antfreq.tofile(outname+'/timestat_antfreq.dat')

