#!/usr/bin/python3

import math
import numpy as np
import sys
import csv

"""
Script for calculating SIS between two spectra csv files, usually the model predictions and the test set targets.
Input as python SIS_similarity.py path_to_file1 path_to_file2
"""

def spectral_information_similarity(spectrum1,spectrum2,conv_matrix,frequencies=list(range(400,4002,2)),threshold=1e-10,std_dev=10):
    length = len(spectrum1)
    nan_mask=np.isnan(spectrum1)+np.isnan(spectrum2)
    # print(length,conv_matrix.shape,spectrum1.shape,spectrum2.shape)
    assert length == len(spectrum2), "compared spectra are of different lengths"
    assert length == len(frequencies), "compared spectra are a different length than the frequencies list, which can be specified"
    spectrum1[spectrum1<threshold]=threshold
    spectrum2[spectrum2<threshold]=threshold
    spectrum1[nan_mask]=0
    spectrum2[nan_mask]=0
    # print(spectrum1.shape,spectrum2.shape)
    spectrum1=np.expand_dims(spectrum1,axis=0)
    spectrum2=np.expand_dims(spectrum2,axis=0)
    # print(spectrum1.shape,spectrum2.shape)
    conv1=np.matmul(spectrum1,conv_matrix)
    # print(conv1[0,1000])
    conv2=np.matmul(spectrum2,conv_matrix)
    conv1[0,nan_mask]=np.nan
    conv2[0,nan_mask]=np.nan
    # print(conv1.shape,conv2.shape)
    sum1=np.nansum(conv1)
    sum2=np.nansum(conv2)
    norm1=conv1/sum1
    norm2=conv2/sum2
    distance=norm1*np.log(norm1/norm2)+norm2*np.log(norm2/norm1)
    sim=1/(1+np.nansum(distance))
    return sim

def import_smiles(file):
    with open(file,'r') as rf:
        r=csv.reader(rf)
        next(r)
        smiles=[]
        for row in r:
            smiles.append(row[0])
        return smiles

def import_data(file):
    with open(file,'r') as rf:
        r=csv.reader(rf)
        next(r)
        data=[]
        for row in r:
            data.append(row)
        return data

def rmse_mae(spectrum1,spectrum2):
    length=len(spectrum1)
    rmse=0.0
    mae=0.0
    for x in range(length):
        rmse+=(spectrum1[x]-spectrum2[x])**2
        mae+=abs(spectrum1[x]-spectrum2[x])
    rmse/=length
    rmse==rmse**0.5
    mae/=length
    return rmse,mae

def make_conv_matrix(frequencies=list(range(400,4002,2)),std_dev=10):
    length=len(frequencies)
    gaussian=[(1/(2*math.pi*std_dev**2)**0.5)*math.exp(-1*((frequencies[i])-frequencies[0])**2/(2*std_dev**2)) for i in range(length)]
    conv_matrix=np.empty([length,length])
    for i in range(length):
        for j in range(length):
            conv_matrix[i,j]=gaussian[abs(i-j)]
    return conv_matrix

def main():
    smiles=import_smiles(sys.argv[1])
    spectra1=import_data(sys.argv[1])
    spectra2=import_data(sys.argv[2])
    sims=[]
    conv_matrix=make_conv_matrix()
    # rmses=[]
    # maes=[]
    for e,mol in enumerate(smiles):
        assert mol==spectra2[e][0], "smiles unmatched or not in order"
        sim=spectral_information_similarity(np.array(spectra1[e][1:],dtype=float),np.array(spectra2[e][1:],dtype=float),conv_matrix)
        # rmse,mae=rmse_mae(np.array(spectra1[e][1:],dtype=float),np.array(spectra2[e][1:],dtype=float))
        sims.append(sim)
        # rmses.append(rmse)
        # maes.append(mae)
        print(f'{e} - {mol}')
    with open('similarity.txt','w') as wf:
        for e,sim in enumerate(sims):
            wf.write(f'{smiles[e]}\t{SIS similarity}\n')

if __name__ == '__main__':
    main()
