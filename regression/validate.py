#_py.validation.dwd.osna..forAdriPasc.py
import numpy as np
import pandas as pd
from pickle import load, dump


def ld(f): return load(open(f, 'rb'))  # _pickle.load
def dp(what, fP): dump(what, open(fP, 'wb'))  # _pickle.dump

# cS = station     # 'wernig',...  weather station code, 6 letters, change accordingly
# cosmoRef = False
# cIni = self.initial_times[0]
# cLea = self.lead_times[1]


# prediction:  deep learning or GLM prediction,  is a np.array
# tista= timestamps, is np.array, corresponds to predictions, same len() as prediction
# cIni=  integer,  current init time,(e.g. 15= 15 UTC)
# cLea=  integer,  current lead time (e.g. 2=  +2 hours)
# cosmoRef= boolean,  if True, it calcs the scores based on pure cosmo output (prediction is ignored in this case)

## Adapted for lists!

def validate(prediction, tista, cIni, cLea, cosmoRef, cS):  # _return scrOut

    #_Load data
    px = '/p/project/deepacf/deeprain/rojascampos1/data/pickles/'
    # _load this data once for each weather station
    obsAndRef = ld(px+f'{cS}.forValid.1x1.y1to7.l1to21.pickle')
    obsAndRefLocl = obsAndRef[obsAndRef['ini'].isin(cIni)]
    obsAndRefLocl = obsAndRefLocl[obsAndRefLocl['lea'].isin(cLea)]
    # obsAndRefLocl = obsAndRef[(obsAndRef.ini == cIni) & (obsAndRef.lea == cLea)]

    loclObs = obsAndRefLocl[obsAndRefLocl.tista.isin(tista)].obs
    verfMons = np.unique([[range(1, 13)[m-2], m, (list(range(13))+[1])[m+1]]
                          for m in pd.unique(pd.to_datetime(tista).month)])
    verfHrs = np.unique([[range(24)[h-1], h, (h+1) % 24]
                         for h in pd.unique(pd.to_datetime(tista).hour)])
    #_Creating obs climatology for obsCDF__
    obsCli = obsAndRef[obsAndRef.tista.dt.hour.isin(verfHrs) & (
        obsAndRef.tista.dt.month.isin(verfMons))][['tista', 'obs']].drop_duplicates()
    obsCli = obsCli[obsCli.obs > 0].obs.values

    # print('obsCli', obsCli)
    # print('obsCli.shape', obsCli.shape)

    obsCdf = np.array([sum(i >= obsCli)/len(obsCli) for i in loclObs])
    if cosmoRef:
        cos = np.array(obsAndRefLocl[obsAndRefLocl.tista.isin(
            pd.to_datetime(tista))].filter(regex='^TOT_P'))  # .iloc[:,3:13])
        cosLEPS = 9999*np.ones(10)
        for iCos in range(10):
            cosCdf = np.array([sum(i >= obsCli)/len(obsCli)
                               for i in cos[:, iCos]])
            cosLEPS[iCos] = abs(cosCdf - obsCdf).mean()
        scrOut = {'LEPS': cosLEPS}
    else:
        prdctCdf = np.array([sum(i >= obsCli)/len(obsCli) for i in prediction])
        scrOut = {'LEPS': abs(prdctCdf - obsCdf).mean()}
    #_End calc LEPS (=the principal score we use, linear error in probability space) __

    #_Contingency table__
    # _threshold list.  Maybe we will modify the thresholds depending on input data and first results
    thrshL = [.25, .5, .75, 1]
    for iT in range(len(thrshL)):
        if cosmoRef:
            FB = ETS = LOR = 9999*np.ones(10)
            for iC in range(10):
                hits = np.sum((cos[:, iC] >= thrshL[iT])
                              & (loclObs >= thrshL[iT]))
                fAls = np.sum((cos[:, iC] >= thrshL[iT])
                              & (loclObs < thrshL[iT]))
                cNgs = np.sum((cos[:, iC] < thrshL[iT])
                              & (loclObs < thrshL[iT]))
                msss = np.sum((cos[:, iC] < thrshL[iT])
                              & (loclObs >= thrshL[iT]))
                # _randomHits, for EquiThrt score
                rHts = (hits+msss)*(hits+fAls)/(cNgs+msss+fAls+hits)
                if hits+msss > 0:
                    FB[iC] = (hits+fAls)/(hits+msss)  # _FreqencyBias
                if hits+msss+fAls+rHts > 0:
                    ETS[iC] = (hits-rHts)/(hits+msss +
                                           fAls+rHts)  # _EquiThrtScore
                if msss*fAls > 0 and hits*cNgs > 0:
                    LOR[iC] = np.log(hits*cNgs/(msss*fAls))  # _LogOddsRatio
            scrOut.update(
                {f'Frequency_Bias_{thrshL[iT]:.2f}': FB, f'Equitable_Threat_Score_{thrshL[iT]:.2f}': ETS, f'Log_Odds_Ratio_{thrshL[iT]:.2f}': LOR})
        else:
            hits = np.sum((prediction >= thrshL[iT]) & (loclObs >= thrshL[iT]))
            fAls = np.sum((prediction >= thrshL[iT]) & (loclObs < thrshL[iT]))
            cNgs = np.sum((prediction < thrshL[iT]) & (loclObs < thrshL[iT]))
            msss = np.sum((prediction < thrshL[iT]) & (loclObs >= thrshL[iT]))
            # _randomHits notig for EquiThrt score
            rHts = (hits+msss)*(hits+fAls)/(cNgs+msss+fAls+hits)
            FB = ETS = LOR = 9999
            if hits+msss > 0:
                FB = (hits+fAls)/(hits+msss)  # _Freqency Bias
            if hits+msss+fAls+rHts > 0:
                ETS = (hits-rHts)/(hits+msss+fAls+rHts)  # _EquiThrt
            if msss*fAls > 0 and hits*cNgs > 0:
                LOR = np.log(hits*cNgs/(msss*fAls))  # _LogOddsRatio
            scrOut.update(
                {f'Frequency_Bias_{thrshL[iT]:.2f}': FB, f'Equitable_Threat_Score_{thrshL[iT]:.2f}': ETS, f'Log_Odds_Ratio_{thrshL[iT]:.2f}': LOR})

    return scrOut  # _def validate(



## works for single events as well as arrays of events. Here y1 would be your model forecast, y2 would be the observation (or the other way round, doesnt matter) and climato is the array with climatology data (probably has to be numpy array).
def leps(y1,y2,climato):
    if len(y1.shape)==0:
        return abs(np.sum(y1>climato) - np.sum(y2>climato))/len(climato)
    else:
        return np.mean([leps(y1[i],y2[i],climato) for i in range(len(y1))])