import numpy as np
import csv
import sklearn.linear_model as lm
import sklearn.metrics as me
from matplotlib import pyplot as plt
import pandas as pd
from pickle import load, dump

def ld(f): return load(open(f, 'rb'))  # _pickle.loadimport pandas as pd

def load_data(path,testtrain,xy):
    outfile=[]
    with open(f'{path}/{testtrain}_{xy}.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            outfile.append(row)
    return np.array(outfile).astype(np.float)

def getpath(station):
    return f"{station}_0_4_5x5_allFeatures_created_2020-10-23"
    


def lasso_specific(station,alpha,extrvars=False,varis=[]):
    xtr=load_data(getpath(station),"train","x")
    ytr=load_data(getpath(station),"train","y")
    if extrvars:
        xtr=extract_vars(xtr,varis)
    lasso1=lm.Lasso(alpha=alpha,max_iter=100000)
    lasso1.fit(xtr,ytr)
    return lasso1

def lasso_general(alpha,extrvars=False,varis=[]):
    stations=["muOsna","braunl","wernig","redlen"]
    xtr=np.row_stack([load_data(getpath(station),"train","x") for station in stations])
    ytr=np.row_stack([load_data(getpath(station),"train","y") for station in stations])
    if extrvars:
        xtr=extract_vars(xtr,varis)
    lasso1=lm.Lasso(alpha=alpha,max_iter=100000)
    lasso1.fit(xtr,ytr)
    return lasso1

def traindata_specific(station,extrvars=False,varis=[]):
    xtr=load_data(getpath(station),"train","x")
    ytr=load_data(getpath(station),"train","y")
    if extrvars:
        xtr=extract_vars(xtr,varis)
    return xtr,ytr

def testdata(station,extrvars=False,varis=[]):
    xte=load_data(getpath(station),"test","x")
    yte=load_data(getpath(station),"test","y")
    if extrvars:
        xte=extract_vars(xtr,varis)
    return xte,yte

def traindata_general(extrvars=False,varis=[]):
    stations=["muOsna","braunl","wernig","redlen"]
    xtr=np.row_stack([load_data(getpath(station),"train","x") for station in stations])
    ytr=np.row_stack([load_data(getpath(station),"train","y") for station in stations])
    if extrvars:
        xtr=extract_vars(xtr,varis)
    return xtr,ytr

def traindata_unobs(station,extrvars=False,varis=[]):
    stations=np.array(["muOsna","braunl","wernig","redlen"])
    i=np.where(stations==station)[0][0]
    xtr=np.row_stack([load_data(getpath(sta),"train","x") for sta in stations[(stations==station)==False]])
    ytr=np.row_stack([load_data(getpath(sta),"train","y") for sta in stations[(stations==station)==False]])
    if extrvars:
        xtr=extract_vars(xtr,varis)
    return xtr,ytr
    
    
    
def lasso(xtr,ytr,alpha):
    lasso1=lm.Lasso(alpha=alpha,max_iter=100000)
    lasso1.fit(xtr,ytr)
    return lasso1

def validate(xte,yte,lassoobj):
    ypre=lassoobj.predict(xte)
    rmse=np.sqrt(me.mean_squared_error(ypre,yte[:,0]))
    nvar=np.sum(abs(lassoobj.coef_)>0.000001)
    return rmse,nvar

def validate_leps(xte,yte,lassoobj,climato):
    ypre=lassoobj.predict(xte)
    lps=leps(ypre,yte[:,0],climato)
    nvar=np.sum(abs(lassoobj.coef_)>0.000001)
    return lps,nvar


def validate_specific(station,alpha,extrvars=False,varis=[]):
    lasso=lasso_specific(station,alpha,extrvars,varis)
    xte=load_data(getpath(station),"test","x")
    yte=load_data(getpath(station),"test","y")
    if extrvars:
        xte=extract_vars(xte,varis)
    ypre=lasso.predict(xte)
    rmse=np.sqrt(me.mean_squared_error(ypre,yte[:,0]))
    nvar=np.sum(abs(lasso.coef_)>0.000001)
    return rmse,nvar
                
def valid_spec_alphas(station,alphas,extrvars=False,varis=[]):
    return np.array([validate_specific(station,a,extrvars,varis) for a in alphas])

def plot_valid(titl,valid,alphas,scor="rmse"):
    plt.scatter(valid[:,1],valid[:,0],c=alphas)
    plt.colorbar()
    plt.title(titl)
    plt.xlabel("N coefs>0")
    plt.ylabel(scor)

    
def extract_1var(x,var):
    return x[:,(np.arange(25)*143)+var]

def extract_vars(x,varis):
    return x[:,np.concatenate([(np.arange(25)*143)+vari for vari in varis])]


def load_climato(station):
    pk=ld(f"/p/scratch/deepacf/deeprain/rojascampos1/data/shared_pickles/{station}.forValid.1x1.y1to7.l1to21.pickle")
    return pk.obs[pk.obs>0.0]

def leps(y1,y2,climato):
    if len(y1.shape)==0:
        return abs(np.sum(np.exp(y1)>climato) - np.sum(np.exp(y2)>climato))/len(climato)
    else:
        return np.mean([leps(y1[i],y2[i],climato) for i in range(len(y1))])


def firstround():
    alphas=np.arange(0.001,0.1,0.001)
    np.save("alphas.npy",alphas)
    for stati in ["muOsna","braunl","wernig","redlen"]:
        cli=load_climato(stati)
### lasso_specific
        xtr,ytr=traindata_specific(stati)
        xte,yte=testdata(stati)
        vali=np.array([validate_leps(xte,yte,lasso(xtr,ytr,a),cli) for a in alphas])
        np.save(f"vali_specific_{stati}_l.npy",vali)
        fig=plt.figure(figsize=(14,10))
        plot_valid(f"{stati} specific",vali,alphas,"leps")
        fig.savefig(f"{stati}_specific_l.png")
        plt.close()
### lasso_general
        xtr,ytr=traindata_general()
        xte,yte=testdata(stati)
        vali=np.array([validate_leps(xte,yte,lasso(xtr,ytr,a),cli) for a in alphas])
        np.save(f"vali_general_{stati}_l.npy",vali)
        fig=plt.figure(figsize=(14,10))
        plot_valid(f"{stati} general",vali,alphas,"leps")
        fig.savefig(f"{stati}_general_l.png")
        plt.close()
### traindata_unobs
        xtr,ytr=traindata_unobs(stati)
        xte,yte=testdata(stati)
        vali=np.array([validate_leps(xte,yte,lasso(xtr,ytr,a),cli) for a in alphas])
        np.save(f"vali_unobs_{stati}_l.npy",vali)
        fig=plt.figure(figsize=(14,10))
        plot_valid(f"{stati} unobserved",vali,alphas,"leps")
        fig.savefig(f"{stati}_unobs_l.png")
        plt.close()

def do_lasso(stati,modus,alpha):
    if modus=="general":
        xtr,ytr=traindata_general()
    if modus=="specific":
        xtr,ytr=traindata_specific(stati)
    if modus=="unobs":
        xtr,ytr=traindata_unobs(stati)
    return lasso(xtr,ytr,alpha)

def secondround():
    stati="braunl"
    mod="specific"
    ls1=do_lasso(stati,mod,0.029)
    np.save(f"boolfilter_lasso_{stati}_{mod}.npy",abs(ls1.coef_)>0.)
    mod="general"
    ls1=do_lasso(stati,mod,0.012)
    np.save(f"boolfilter_lasso_{stati}_{mod}.npy",abs(ls1.coef_)>0.)
    mod="unobs"
    ls1=do_lasso(stati,mod,0.016)
    np.save(f"boolfilter_lasso_{stati}_{mod}.npy",abs(ls1.coef_)>0.)
    stati="muOsna"
    mod="specific"
    ls1=do_lasso(stati,mod,0.015)
    np.save(f"boolfilter_lasso_{stati}_{mod}.npy",abs(ls1.coef_)>0.)
    mod="general"
    ls1=do_lasso(stati,mod,0.01)
    np.save(f"boolfilter_lasso_{stati}_{mod}.npy",abs(ls1.coef_)>0.)
    mod="unobs"
    ls1=do_lasso(stati,mod,0.019)
    np.save(f"boolfilter_lasso_{stati}_{mod}.npy",abs(ls1.coef_)>0.)
    stati="wernig"
    mod="specific"
    ls1=do_lasso(stati,mod,0.011)
    np.save(f"boolfilter_lasso_{stati}_{mod}.npy",abs(ls1.coef_)>0.)
    mod="general"
    ls1=do_lasso(stati,mod,0.006)
    np.save(f"boolfilter_lasso_{stati}_{mod}.npy",abs(ls1.coef_)>0.)
    mod="unobs"
    ls1=do_lasso(stati,mod,0.01)
    np.save(f"boolfilter_lasso_{stati}_{mod}.npy",abs(ls1.coef_)>0.)
    stati="redlen"
    mod="specific"
    ls1=do_lasso(stati,mod,0.05)
    np.save(f"boolfilter_lasso_{stati}_{mod}.npy",abs(ls1.coef_)>0.)
    mod="general"
    ls1=do_lasso(stati,mod,0.024)
    np.save(f"boolfilter_lasso_{stati}_{mod}.npy",abs(ls1.coef_)>0.)
    mod="unobs"
    ls1=do_lasso(stati,mod,0.025)
    np.save(f"boolfilter_lasso_{stati}_{mod}.npy",abs(ls1.coef_)>0.)


def thirdround():
    mod="specific"
    for stati in ["braunl","muOsna","wernig","redlen"]:
        xtr,ytr=traindata_specific(stati)
        fil=np.load(f"boolfilter_lasso_{stati}_{mod}.npy")
        xtr=xtr[:,fil]
        las=lasso(xtr,ytr,0.)
        xte,yte=testdata(stati)
        ypre=las.predict(xte[:,fil])
        np.save(f"predicted_precipitation_{stati}_{mod}.npy",ypre)
    mod="general"
    for stati in ["braunl","muOsna","wernig","redlen"]:
        xtr,ytr=traindata_general()
        fil=np.load(f"boolfilter_lasso_{stati}_{mod}.npy")
        xtr=xtr[:,fil]
        las=lasso(xtr,ytr,0.)
        xte,yte=testdata(stati)
        ypre=las.predict(xte[:,fil])
        np.save(f"predicted_precipitation_{stati}_{mod}.npy",ypre)
    mod="unobs"
    for stati in ["braunl","muOsna","wernig","redlen"]:
        xtr,ytr=traindata_unobs(stati)
        fil=np.load(f"boolfilter_lasso_{stati}_{mod}.npy")
        xtr=xtr[:,fil]
        las=lasso(xtr,ytr,0.)
        xte,yte=testdata(stati)
        ypre=las.predict(xte[:,fil])
        np.save(f"predicted_precipitation_{stati}_{mod}.npy",ypre)


def thirdround2():
    mod="specific"
    for stati in ["braunl","muOsna","wernig","redlen"]:
        xtr,ytr=traindata_specific(stati)
        fil=np.load(f"boolfilter_lasso_{stati}_{mod}.npy")
        xtr=xtr[:,fil]
        las=lm.TweedieRegressor(power=0,alpha=0,link="identity",max_iter=100000)
        las.fit(xtr,ytr)
        xte,yte=testdata(stati)
        ypre=las.predict(xte[:,fil])
        np.save(f"predicted_precipitation_{stati}_{mod}2.npy",ypre)
    mod="general"
    for stati in ["braunl","muOsna","wernig","redlen"]:
        xtr,ytr=traindata_general()
        fil=np.load(f"boolfilter_lasso_{stati}_{mod}.npy")
        xtr=xtr[:,fil]
        las=lm.TweedieRegressor(power=0,alpha=0,link="identity",max_iter=100000)
        las.fit(xtr,ytr)
        xte,yte=testdata(stati)
        ypre=las.predict(xte[:,fil])
        np.save(f"predicted_precipitation_{stati}_{mod}2.npy",ypre)
    mod="unobs"
    for stati in ["braunl","muOsna","wernig","redlen"]:
        xtr,ytr=traindata_unobs(stati)
        fil=np.load(f"boolfilter_lasso_{stati}_{mod}.npy")
        xtr=xtr[:,fil]
        las=lm.TweedieRegressor(power=0,alpha=0,link="identity",max_iter=100000)
        las.fit(xtr,ytr)
        xte,yte=testdata(stati)
        ypre=las.predict(xte[:,fil])
        np.save(f"predicted_precipitation_{stati}_{mod}2.npy",ypre)

def thirdround_gordon():
    mod="specific"
    for stati in ["braunl","muOsna","wernig","redlen"]:
        xtr,ytr=traindata_specific(stati)
        fil=load_gordons_lasso(stati)
        xtr=xtr[:,fil]
        ytr=ytr[:,0]
        las=lm.TweedieRegressor(power=0,alpha=0,link="identity",max_iter=100000)
        las.fit(xtr,ytr)
        xte,yte=testdata(stati)
        ypre=las.predict(xte[:,fil])
        np.save(f"predictions_IDgordon/predicted_precipitation_{stati}_{mod}2.npy",ypre)
    mod="general"
    for stati in ["braunl","muOsna","wernig","redlen"]:
        xtr,ytr=traindata_general()
        fil=load_gordons_lasso(stati)
        xtr=xtr[:,fil]
        ytr=ytr[:,0]
        las=lm.TweedieRegressor(power=0,alpha=0,link="identity",max_iter=100000)
        las.fit(xtr,ytr)
        xte,yte=testdata(stati)
        ypre=las.predict(xte[:,fil])
        np.save(f"predictions_IDgordon/predicted_precipitation_{stati}_{mod}2.npy",ypre)
    mod="unobs"
    for stati in ["braunl","muOsna","wernig","redlen"]:
        xtr,ytr=traindata_unobs(stati)
        fil=load_gordons_lasso(stati)
        xtr=xtr[:,fil]
        ytr=ytr[:,0]
        las=lm.TweedieRegressor(power=0,alpha=0,link="identity",max_iter=100000)
        las.fit(xtr,ytr)
        xte,yte=testdata(stati)
        ypre=las.predict(xte[:,fil])
        np.save(f"predictions_IDgordon/predicted_precipitation_{stati}_{mod}2.npy",ypre)

def thirdround3():
    mod="specific"
#    for stati in ["braunl","muOsna","wernig","redlen"]:
#        xtr,ytr=traindata_specific(stati)
#        fil=np.load(f"boolfilter_lasso_{stati}_{mod}.npy")
#        xtr=xtr[:,fil]
#        las=lm.TweedieRegressor(power=2,alpha=0,link="log",max_iter=100000)
#        las.fit(xtr,np.exp(ytr))
#        xte,yte=testdata(stati)
#        ypre=las.predict(xte[:,fil])
#        np.save(f"predicted_precipitation_{stati}_{mod}3.npy",ypre)
#    mod="general"
#    for stati in ["braunl","muOsna","wernig","redlen"]:
#        xtr,ytr=traindata_general()
#        fil=np.load(f"boolfilter_lasso_{stati}_{mod}.npy")
#        xtr=xtr[:,fil]
#        las=lm.TweedieRegressor(power=2,alpha=0,link="log",max_iter=100000)
#        las.fit(xtr,np.exp(ytr))
#        xte,yte=testdata(stati)
#        ypre=las.predict(xte[:,fil])
#        np.save(f"predicted_precipitation_{stati}_{mod}3.npy",ypre)
    mod="unobs"
    for stati in ["braunl","muOsna","wernig","redlen"]:
        xtr,ytr=traindata_unobs(stati)
        fil=np.load(f"boolfilter_lasso_{stati}_{mod}.npy")
        xtr=xtr[:,fil]
        las=lm.TweedieRegressor(power=2,alpha=0,link="log",max_iter=100000)
        las.fit(xtr,np.exp(ytr))
        xte,yte=testdata(stati)
        ypre=las.predict(xte[:,fil])
        np.save(f"predicted_precipitation_{stati}_{mod}3.npy",ypre)


def firstround_leps():
    alphas=np.arange(0.001,0.1,0.001)
    np.save("alphas.npy",alphas)
    for stati in ["muOsna","braunl","wernig","redlen"]:
### lasso_specific
        clima=load_climato(stati)
        xtr,ytr=traindata_specific(stati)
        xte,yte=testdata(stati)
        vali=np.array([validate(xte,yte,lasso(xtr,ytr,a)) for a in alphas])
        np.save(f"vali_specific_{stati}_l.npy",vali)
        fig=plt.figure(figsize=(14,10))
        plot_valid(f"{stati} specific",vali,alphas)
        fig.savefig(f"{stati}_specific_l.png")
        plt.close()
### lasso_general
        xtr,ytr=traindata_general()
        xte,yte=testdata(stati)
        vali=np.array([validate(xte,yte,lasso(xtr,ytr,a)) for a in alphas])
        np.save(f"vali_general_{stati}_l.npy",vali)
        fig=plt.figure(figsize=(14,10))
        plot_valid(f"{stati} general",vali,alphas)
        fig.savefig(f"{stati}_general_l.png")
        plt.close()
### traindata_unobs
        xtr,ytr=traindata_unobs(stati)
        xte,yte=testdata(stati)
        vali=np.array([validate(xte,yte,lasso(xtr,ytr,a)) for a in alphas])
        np.save(f"vali_unobs_{stati}_l.npy",vali)
        fig=plt.figure(figsize=(14,10))
        plot_valid(f"{stati} unobserved",vali,alphas)
        fig.savefig(f"{stati}_unobs_l.png")
        plt.close()


def print_validate(station,modus):
    print(" ")
    alphas=np.load("alphas.npy")
    v1=np.load(f"vali_{modus}_{station}_l.npy") 
    print(f"alpha selection --- {station} {modus}")
    print(" ")
    print("alpha  |   LEPS     |   N_coef>0")
    print(" ")
    for i in np.argsort(v1[:,0])[:10]:
        print(np.round(alphas[i],7)," | ",np.round(v1[i,0],6),"   | ",v1[i,1])


def load_classy(station,testrain,xy):
    return np.load(f"/p/scratch/deepacf/deeprain/rojascampos1/data/shared_rain_norain/{station}/{testrain}_{xy}.npy")


def validate_loreg(xte,yte,lassoobj):
    forc=lassoobj.predict(xte)
    accuracy=np.sum(forc==(yte>=0.1))/len(yte)
    nvar=np.sum(abs(lassoobj.coef_)>0.00001)
    print(accuracy,nvar)
    return accuracy,nvar

def loreg(xtr,ytr,alpha,penal,mxi=1000):
    lo=lm.LogisticRegression(C=1/alpha,penalty=penal,solver="saga",max_iter=mxi)
    lo.fit(xtr,ytr>=0.1)
    return lo


def do_lo_lasso(station):
    alphas=[8,16,32,64]
    xtr=load_classy(station,'trn','x')
    ytr=load_classy(station,'trn','y')
    xte=load_classy(station,'tst','x')
    yte=load_classy(station,'tst','y')
    vali=np.array([validate_loreg(xte,yte,loreg(xtr,ytr,a,"l1")) for a in alphas])
    return vali


def lo_lasso_splitsets(station):
    for splitset in [[0,20],[20,40],[40,60],[60,80],[80,100],[100,120],[120,143]]:
        alphas=[1,2,4,8,16,32,64,128]
        xtr=extract_vars(load_classy(station,'trn','x'),np.arange(splitset[0],splitset[1]))
        ytr=load_classy(station,'trn','y')
        xte=extract_vars(load_classy(station,'tst','x'),np.arange(splitset[0],splitset[1]))
        yte=load_classy(station,'tst','y')
        vali=np.array([validate_loreg(xte,yte,loreg(xtr,ytr,a,"l1",mxi=10000)) for a in alphas])
        np.save(f"lo_lasso/vali_lo_{station}_splitset_{splitset[0]}_{splitset[1]}.npy",vali)


def load_gordons_lasso(station):
    return np.loadtxt(f"{station}_ID_Predictors.txt",delimiter=",",dtype="int")-1


def compare(station,modus):
    y1=np.load(f"predictions_IDgordon/predicted_precipitation_{station}_{modus}.npy")
    y2=np.load(f"predicted_precipitation_{station}_{modus}2.npy")
    xte,yte=testdata(station)
    cli=load_climato(station)
    yte=yte[:,0]
    lps1=leps(y2,yte,cli)
    lps2=leps(y1,yte,cli)
    print("leps lasso martin:")
    print(lps1)
    print("leps lasso gordon:")
    print(lps2)


def do_loreg(stati,featuresel):
    alphas=[0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]
    xtr=load_classy(stati,"trn","x")
    ytr=load_classy(stati,"trn","y")
    xte=load_classy(stati,"tst","x")
    yte=load_classy(stati,"tst","y")
    accuracies=[]
    if featuresel=="gordon":
        xtr=xtr[:,load_gordons_lasso(stati)]
        xte=xte[:,load_gordons_lasso(stati)]
    if featuresel=="martin":
        bf=np.load(f"boolfilter_lasso_{stati}_specific.npy")
        xtr=xtr[:,bf]
        xte=xte[:,bf]
    for al in alphas:
        lr=loreg(xtr,ytr,al,"l2",mxi=10000)
        forc=lr.predict(xte)
        accuracies.append(np.sum(forc==(yte>=0.1))/len(yte))
        print(al)
        np.save(f"loregaccuracies_{stati}_{featuresel}.npy",accuracies)

def do_loreg2(stati,featuresel,alphas):
    xtr=load_classy(stati,"trn","x")
    ytr=load_classy(stati,"trn","y")
    xte=load_classy(stati,"tst","x")
    yte=load_classy(stati,"tst","y")
    accuracies=[]
    if featuresel=="gordon":
        xtr=xtr[:,load_gordons_lasso(stati)]
        xte=xte[:,load_gordons_lasso(stati)]
    if featuresel=="martin":
        bf=np.load(f"boolfilter_lasso_{stati}_specific.npy")
        xtr=xtr[:,bf]
        xte=xte[:,bf]
    for al in alphas:
        lr=loreg(xtr,ytr,al,"l2",mxi=10000)
        forc=lr.predict(xte)
        accuracies.append(np.sum(forc==(yte>=0.1))/len(yte))
        print(al)
        np.save(f"loregaccuracies_{stati}_{featuresel}2.npy",[accuracies,alphas])

        
def do_loreg3(stati,featuresel,alpha):
    xtr=load_classy(stati,"trn","x")
    ytr=load_classy(stati,"trn","y")
    xte=load_classy(stati,"tst","x")
    yte=load_classy(stati,"tst","y")
    accuracies=[]
    if featuresel=="gordon":
        xtr=xtr[:,load_gordons_lasso(stati)]
        xte=xte[:,load_gordons_lasso(stati)]
    if featuresel=="martin":
        bf=np.load(f"boolfilter_lasso_{stati}_specific.npy")
        xtr=xtr[:,bf]
        xte=xte[:,bf]
    lr=loreg(xtr,ytr,alpha,"l2",mxi=10000)
    forc=lr.predict(xte)
    np.save(f"logisticregressionpredictions_{stati}_{featuresel}.npy",forc)

