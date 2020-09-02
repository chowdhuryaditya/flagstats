import numpy as np
import matplotlib.pyplot as plt
from __casac__ import *
import os
from astropy.time import Time
import astropy.units as u
from matplotlib.backends.backend_pdf import PdfPages

def setFonts(SMALL_SIZE = 18,MEDIUM_SIZE = 24,BIGGER_SIZE = 30,axisLW=3,ticksize=10):
	plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the xtick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the ytick labels
	plt.rc('xtick.major', pad=12,size=ticksize)  #size of x major ticks   
	plt.rc('ytick.major', pad=12,size=ticksize)  #size of y major ticks
	plt.rc('xtick.minor', size=ticksize/2)       #size of x minor ticks
	plt.rc('ytick.minor', size=ticksize/2)        #size of y minor ticks
	plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
	#plt.rc('figure', fontsize=BIGGER_SIZE)  # fontsize of the figure title

setFonts(SMALL_SIZE = 15,MEDIUM_SIZE = 15,BIGGER_SIZE = 15,axisLW=5,ticksize=9)



antenna=['C00','C01','C02','C03','C04','C05','C06','C08','C09','C10','C11','C12','C13','C14','E02', 'E03','E04','E05','E06','S01','S02','S03','S04','S06','W01','W02','W03','W04','W05','W06']

nant=30


def getantflags(blflags,ant1,ant2):	
	blmask=np.zeros(len(blflags))
	nant=30
	antflags=np.zeros(nant)
	for iant in range(nant):	
		mask=np.logical_or(ant1==iant,ant2==iant)
		datasel=blflags[mask]
		antflags[iant]=np.mean(datasel)
		if(antflags[iant]==1.0):
			blmask[mask]=1.0
	return antflags,blmask,np.sum(antflags==1.0)

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

def getdeadant(time,obsid,timestat_bl,excludecsqrbl=False):

	ant1,ant2=getantindx()
	antindx=np.zeros(len(ant1))
	uniqoid,oidcounts=np.unique(obsid,return_counts=True)
	deadflagcounts=np.zeros(timestat_bl.shape[1])
	blmask=np.zeros((0,timestat_bl.shape[1]),dtype=np.uint64)
	ndeadant=np.zeros(len(uniqoid))
	nrecords=len(obsid)
	for i in range(0,len(uniqoid)):
		indx=obsid==uniqoid[i]
		blflags=np.mean(timestat_bl[indx,:],axis=0)
		antflags,blmask_,ndeadant[i]=getantflags(blflags,ant1,ant2)
		deadflagcounts+=blmask_*oidcounts[i]
		blmask=np.append(blmask,np.repeat(np.expand_dims(blmask_,axis=0),oidcounts[i],axis=0),axis=0)
	if(excludecsqrbl):
		ibl=0
		for iant1 in range(nant):
			for iant2 in range(iant1+1,nant):
				if(iant1<14 and iant2<14):
					deadflagcounts[ibl]=nrecords
					blmask[:,ibl]=1
				ibl+=1
	return deadflagcounts,1-blmask,uniqoid,ndeadant

def getFrequencyDist(chanstat,deadflagcounts,nrecords):
	ant1,ant2=getantindx()
	chanstat*=nrecords
	antwisestat=np.zeros((nant+1,chanstat.shape[0]))
	for iant in range(nant):
		mask=np.logical_or(ant1==iant,ant2==iant)
		corr=np.mean(deadflagcounts[mask])
		if(corr==nrecords):
			antwisestat[iant+1,:]=1.0
		else:
			antwisestat[iant+1,:]=(np.mean(chanstat[:,mask],axis=1)-corr)/(nrecords-corr)
	corr=np.mean(deadflagcounts)
	antwisestat[0,:]=(np.mean(chanstat,axis=1)-corr)/(nrecords-corr)
	return antwisestat

def getTimeDist(timestat_bl,blmask,navg=1):
	ant1,ant2=getantindx()
	ntime=timestat_bl.shape[0]
	antwisestat=np.zeros((nant+1,ntime))
	maskedstat=timestat_bl*blmask
	for iant in range(nant):
		indx=np.logical_or(ant1==iant,ant2==iant)
		nbl=np.sum(blmask[:,indx],axis=1)			
		antwisestat[iant+1,:]=np.sum(maskedstat[:,indx],axis=1)/nbl
		antwisestat[iant+1,nbl==0]=1.0


	antwisestat[0,:]=np.sum(maskedstat,axis=1)/np.sum(blmask,axis=1)
	navgrec=ntime//navg
	antwisestat=np.mean(antwisestat[:,:navgrec*navg].reshape(nant+1,navgrec,navg),axis=2)
	return antwisestat

def getTimeFreqDist(timestat_freq,blmask,navg=1):
	timefreqdist=(timestat_freq*blmask.shape[1]-np.sum(1-blmask,axis=1,keepdims=True))/np.sum(blmask,axis=1,keepdims=True)
	nrec=timefreqdist.shape[0]
	nrecavg=nrec//navg
	timefreqdist=np.mean(timefreqdist.T[:,:nrecavg*navg].reshape(timefreqdist.shape[1],nrecavg,navg),axis=2)
	return timefreqdist

def getTimeFreqDistAnt(timestat_antfreq,blmask,navg=1):
	ant1,ant2=getantindx()
	
	for iant in range(0,nant):
		mask=np.logical_or(ant1==iant,ant2==iant)
		nbl=np.sum(blmask[:,mask],axis=1)			
		timestat_antfreq[iant]=(timestat_antfreq[iant]*np.sum(mask)-np.sum(1-blmask[:,mask],axis=1))/nbl
		timestat_antfreq[iant,:,nbl==0]=1.0
	
	nrec=timestat_antfreq.shape[2]
	nrecavg=nrec//navg
	timestat_antfreq=np.mean(timestat_antfreq[:,:,:nrecavg*navg].reshape(timestat_antfreq.shape[0],timestat_antfreq.shape[1],nrecavg,navg),axis=3)
	return timestat_antfreq

def plottimedist(time,timestamp,obsid,timedist,outfile):
	uniqobsid=np.unique(obsid)
	timestamp=(timestamp+4.5)
	#timestamp[timestamp>=24]-=24

	pdf=PdfPages(outfile)
	tlabels=np.arange(0,17,2)+16
	tlabels[tlabels>=24]-=24
	for iant in range(nant+1):
		fig=plt.figure(figsize=(12,7))
		if(iant==0):
			plt.title('All antenna')
		else:
			plt.title('%d : %s'%(iant,antenna[iant-1]))
		for iobsid in range(0,len(uniqobsid)):
			indx=(obsid==uniqobsid[iobsid])		
			stime=time[indx][0]
			plt.plot(timestamp[indx][:-2]-16,timedist[iant,indx][:-2],label='%d/%d/%d'%(stime.day,stime.month,stime.year),lw=2,marker='o',ms=2)
		plt.legend(ncol=4)
		plt.xticks(np.arange(0,17,2),tlabels)
		plt.ylim((0,1.05))
		plt.ylabel("Flagged Fraction")
		plt.xlabel("Time (hr, IST)")
		plt.minorticks_on()

		pdf.savefig(fig)
		plt.clf()
		plt.close()
	pdf.close()

def plotfreqdist(freq,frequencydist,indxlist,outfile):
	pdf=PdfPages(outfile)
	for iant in range(nant+1):
		fig=plt.figure(figsize=(12,7))
		if(iant==0):
			plt.title('All antenna')
		else:
			plt.title('%d : %s'%(iant,antenna[iant-1]))
		for i in range(0,len(indxlist[iant])):
			plt.axvspan(freq[indxlist[iant][i,0]],freq[indxlist[iant][i,1]],facecolor='r', alpha=0.5,edgecolor='r')
		plt.subplots_adjust(bottom=0.2,top=0.9)
		plt.plot(freq,frequencydist[iant],lw=2,c='blue')
		plt.ylim((0,1.05))
		plt.ylabel("Flagged Fraction")
		plt.xlabel("Frequency (MHz)")
		plt.minorticks_on()
		pdf.savefig(fig)
		plt.clf()
		plt.close()
	
	pdf.close()

	
def plottimefreqdist(time,timestamp,obsbreakindx,freq,timefreqdist,timefreqdistAnt,outfile):
	pdf=PdfPages(outfile)
	ext=(0,len(time),freq[0],freq[-1])
	timestamp=(timestamp+4.5)
	timestamp[timestamp>=24]-=24
	ticks=np.linspace(0,len(time)-2,10).astype("int")
	tsticks=timestamp[ticks]
	labels=[]
	for i in range(0,len(ticks)):
		labels.append('%.1f'%tsticks[i])
	for iant in range(nant+1):
		fig=plt.figure(figsize=(12,7))
		if(iant==0):
			plt.title('All antenna')
			plt.imshow(timefreqdist,origin='lower',aspect='auto',vmin=0,vmax=1,extent=ext)
		else:
			plt.title('%d : %s'%(iant,antenna[iant-1]))
			plt.imshow(timefreqdistAnt[iant-1],origin='lower',aspect='auto',vmin=0,vmax=1,extent=ext)
		for i in range(0,len(obsbreakindx)-1):
			plt.axvline(obsbreakindx[i],c='white',lw=6)
		plt.subplots_adjust(bottom=0.2,top=0.9)
		plt.ylabel("Frequency (MHz)")
		plt.xlabel("Time (hr, IST)")
		plt.xticks(ticks,labels)
		cbar=plt.colorbar()
		cbar.set_label('Flagged Fraction')
		#plt.minorticks_on()
		pdf.savefig(fig)
		plt.clf()
		plt.close()
	
	pdf.close()

def gettimestamps(time):
	tstamp=np.zeros(len(time))
	for i in range(0,len(tstamp)):
		tstamp[i]=time[i].hour+time[i].minute/60.0+(time[i].second+time[i].microsecond*1e-6)/3600.0
	return tstamp


def getobsbreakindx(obsid):
	uniqobsid=np.unique(obsid)
	breakindx=np.zeros(len(uniqobsid))
	indx=np.arange(len(obsid))
	for i in range(len(uniqobsid)):
		mask=obsid==uniqobsid[i]
		breakindx[i]=indx[mask][-1]
	return breakindx

def medfilt(x,window):
	xs=np.zeros(x.shape)
	n=len(x)
	for i in range(0,n):
		start=i-window//2
		stop=i+window//2
		if(start<0):
			start=0
		if(stop>=n):
			stop=n-1
		xs[i]=np.median(x[start:stop])
	return xs

def getfreqlist(ant,mask,freq):
	lst=[]
	indxlst=[]
	strt=0
	for i in range(0,len(mask)-1):
		if(mask[i] and not mask[i+1]):
			lst.append([ant,freq[strt],freq[i]])
			indxlst.append((strt,i+1))
		if(not mask[i] and mask[i+1]):
			strt=i+1
	lst=np.array(lst)
	indxlst=np.array(indxlst)
	if(len(lst)==0):
		return np.zeros((0,3)),np.zeros((0,2))		
	if(lst[0,1]==freq[0]):
		lst=lst[1:]
		indxlst=indxlst[1:]
	if(lst[-1,2]==freq[-1]):
		lst=lst[:-1]
		indxlst=indxlst[:-1]
	return lst,indxlst

def getnarrowbandlist(freq,freqdist,smoothwindow,threshold):
	freqdistsmooth=np.zeros(freqdist.shape)	
	freqlist=np.zeros((0,3))
	indxlist=[]
	for i in range(0,freqdist.shape[0]):
		freqdistsmooth[i]=medfilt(freqdist[i],smoothwindow)
		diff=freqdist[i]-freqdistsmooth[i]
		mask=diff>threshold
		freqlist_,indxlist_=getfreqlist(i,mask,freq)
		freqlist=np.append(freqlist,freqlist_,axis=0)
		indxlist.append(indxlist_)
	return indxlist,freqlist

def writediagnostics(statfolder,freq,frequencydist,freqlist):
	header='1:Frequency (MHz) 2:All Antenna '
	header2='Antenna ID 0 represents all antennas\nID 1 through 30:\n'
	fmt='%.6f %.3f '
	antstr=np.empty(freqlist.shape[0],dtype='S6')
	antstr[freqlist[:,0]==0]='AllAnt'
	for i in range(0,nant):
		header+='%d:%s '%(i+3,antenna[i])
		header2+='%d:%s '%(i+1,antenna[i])
		if(i>0 and i%6==0):
			header2+='\n'
		fmt+='%.3f '
		antstr[freqlist[:,0]==(i+1)]==antenna[i]
	header2+='\n\nAntenna_ID start_frequency(MHz) stop_frequency(MHz)'
	np.savetxt(statfolder+'/freqdist.dat',np.append(np.expand_dims(freq,axis=0),frequencydist,axis=0).T,fmt=fmt,header=header)
	np.savetxt(statfolder+'/rfilist.dat',freqlist,fmt='%d %.4f %.4f',header=header2)

	
	

def getdiagnostics(statfolder,excludecsqrbl=False,smoothwindow=90,threshold=0.01,navg=12):

	chanstat=np.loadtxt(statfolder+'/chanstat.dat')
	timestat_bl=np.loadtxt(statfolder+'/timestat_bl.dat')
	timestat_freq=np.loadtxt(statfolder+'/timestat_freq.dat')
	nrecords=timestat_bl.shape[0]
	nchan=timestat_freq.shape[1]
	#timestat_antfreq=np.fromfile(statfolder+'/timestat_antfreq.dat').reshape((nant,nchan,nrecords))
	freq=np.loadtxt(statfolder+'/freq.dat')
	time,scan_number,obsid=np.loadtxt(statfolder+'/scaninfo.dat',unpack=True)
	
	deadflagcounts,blmask,uniqoid,ndeadant=getdeadant(time,obsid,timestat_bl,excludecsqrbl=excludecsqrbl)
	frequencydist=getFrequencyDist(chanstat,deadflagcounts,nrecords)
	indxlist,freqlist=getnarrowbandlist(freq,frequencydist,smoothwindow=smoothwindow,threshold=threshold)
	
	
	timedist=getTimeDist(timestat_bl,blmask,navg=navg)
	timefreqdist=getTimeFreqDist(timestat_freq,blmask,navg=navg)
	#timefreqdistAnt=getTimeFreqDistAnt(timestat_antfreq,blmask,navg=navg)

	time=time[:navg*(nrecords//navg):navg]
	obsid=obsid[:navg*(nrecords//navg):navg]
	obsbreakindex=getobsbreakindx(obsid)
	timeastropy = Time(time*u.s, format='mjd').datetime
	timestamp=gettimestamps(timeastropy)
	writediagnostics(statfolder,freq,frequencydist,freqlist)
	plt.ioff()
	plotfreqdist(freq,frequencydist,indxlist,outfile=statfolder+'/freqdist.pdf')
	plottimedist(timeastropy,timestamp,obsid,timedist,outfile=statfolder+'/timedist.pdf')
	#plottimefreqdist(time,timestamp,obsbreakindex,freq,timefreqdist,timefreqdistAnt,outfile=statfolder+'/timefreqdist.pdf')
	plt.ion()
