from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from tqdm import trange

def sid(model_spectra: torch.tensor, target_spectra: torch.tensor, threshold: float = 1e-8, eps: float = 1e-8, torch_device: str = 'cpu') -> torch.tensor:
    # normalize the model spectra before comparison
    nan_mask=torch.isnan(target_spectra)+torch.isnan(model_spectra)
    nan_mask=nan_mask.to(device=torch_device)
    zero_sub=torch.zeros_like(target_spectra,device=torch_device)
    model_spectra = model_spectra.to(torch_device)
    model_spectra[model_spectra < threshold] = threshold
    sum_model_spectra = torch.sum(torch.where(nan_mask,zero_sub,model_spectra),axis=1)
    sum_model_spectra = torch.unsqueeze(sum_model_spectra,axis=1)
    model_spectra = torch.div(model_spectra,sum_model_spectra)
    # calculate loss value
    if not isinstance(target_spectra,torch.Tensor):
        target_spectra = torch.tensor(target_spectra)
    target_spectra = target_spectra.to(torch_device)
    loss = torch.ones_like(target_spectra)
    loss = loss.to(torch_device)
    target_spectra[nan_mask]=1
    model_spectra[nan_mask]=1
    loss = torch.mul(torch.log(torch.div(model_spectra,target_spectra)),model_spectra) \
        + torch.mul(torch.log(torch.div(target_spectra,model_spectra)),target_spectra)
    loss[nan_mask]=0
    loss = torch.sum(loss,axis=1)
    return loss

def jsd(model_spectra: torch.tensor, target_spectra: torch.tensor, threshold: float = 1e-8, eps: float = 1e-8, torch_device: str = 'cpu') -> torch.tensor:
    # normalize the model spectra before comparison
    nan_mask=torch.isnan(target_spectra)+torch.isnan(model_spectra)
    nan_mask=nan_mask.to(device=torch_device)
    zero_sub=torch.zeros_like(target_spectra,device=torch_device)
    model_spectra = model_spectra.to(torch_device)
    model_spectra[model_spectra < threshold] = threshold
    sum_model_spectra = torch.sum(torch.where(nan_mask,zero_sub,model_spectra),axis=1)
    sum_model_spectra = torch.unsqueeze(sum_model_spectra,axis=1)
    model_spectra = torch.div(model_spectra,sum_model_spectra)
    # average spectra
    if not isinstance(target_spectra,torch.Tensor):
        target_spectra = torch.tensor(target_spectra)
    target_spectra = target_spectra.to(torch_device)
    target_spectra[nan_mask]=1
    model_spectra[nan_mask]=1
    avg_spectra = torch.ones_like(target_spectra)
    avg_spectra = avg_spectra.to(torch_device)
    avg_spectra = torch.add(target_spectra,model_spectra)
    avg_spectra = torch.div(avg_spectra,2)
    # calculate loss
    loss = torch.ones_like(target_spectra)
    loss = loss.to(torch_device)
    loss = torch.mul(torch.log(torch.div(model_spectra,avg_spectra)),model_spectra) \
        + torch.mul(torch.log(torch.div(target_spectra,avg_spectra)),target_spectra)
    loss[nan_mask]=0
    loss = torch.div(torch.sum(loss,axis=1),2)
    return loss

def stmse(model_spectra: torch.tensor, target_spectra: torch.tensor, threshold: float = 1e-8, eps: float = 1e-8, torch_device: str = 'cpu') -> torch.tensor:
    # normalize the model spectra before comparison
    nan_mask=torch.isnan(target_spectra)+torch.isnan(model_spectra)
    nan_mask=nan_mask.to(device=torch_device)
    zero_sub=torch.zeros_like(target_spectra,device=torch_device)
    model_spectra = model_spectra.to(torch_device)
    model_spectra[model_spectra < threshold] = threshold
    sum_model_spectra = torch.sum(torch.where(nan_mask,zero_sub,model_spectra),axis=1)
    sum_model_spectra = torch.unsqueeze(sum_model_spectra,axis=1)
    model_spectra = torch.div(model_spectra,sum_model_spectra)
    # calculate loss value
    if not isinstance(target_spectra,torch.Tensor):
        target_spectra = torch.tensor(target_spectra)
    target_spectra = target_spectra.to(torch_device)
    loss = torch.ones_like(target_spectra)
    loss = loss.to(torch_device)
    target_spectra[nan_mask]=1
    model_spectra[nan_mask]=1
    loss = torch.mean(torch.div((model_spectra-target_spectra)**2,target_spectra),dim=1)
    return loss

def srmse(model_spectra: torch.tensor, target_spectra: torch.tensor, threshold: float = 1e-8, eps: float = 1e-8, torch_device: str = 'cpu') -> torch.tensor:
    # normalize the model spectra before comparison
    nan_mask=torch.isnan(target_spectra)+torch.isnan(model_spectra)
    nan_mask=nan_mask.to(device=torch_device)
    zero_sub=torch.zeros_like(target_spectra,device=torch_device)
    model_spectra = model_spectra.to(torch_device)
    model_spectra[model_spectra < threshold] = threshold
    sum_model_spectra = torch.sum(torch.where(nan_mask,zero_sub,model_spectra),axis=1)
    sum_model_spectra = torch.unsqueeze(sum_model_spectra,axis=1)
    model_spectra = torch.div(model_spectra,sum_model_spectra)
    # calculate loss value
    if not isinstance(target_spectra,torch.Tensor):
        target_spectra = torch.tensor(target_spectra)
    target_spectra = target_spectra.to(torch_device)
    loss = torch.ones_like(target_spectra)
    loss = loss.to(torch_device)
    target_spectra[nan_mask]=1
    model_spectra[nan_mask]=1
    loss = torch.mean((model_spectra-target_spectra)**2,dim=1)
    loss = torch.sqrt(loss + eps)
    return loss

def smse(model_spectra: torch.tensor, target_spectra: torch.tensor, threshold: float = 1e-8, eps: float = 1e-8, torch_device: str = 'cpu') -> torch.tensor:
    # normalize the model spectra before comparison
    nan_mask=torch.isnan(target_spectra)+torch.isnan(model_spectra)
    nan_mask=nan_mask.to(device=torch_device)
    zero_sub=torch.zeros_like(target_spectra,device=torch_device)
    model_spectra = model_spectra.to(torch_device)
    model_spectra[model_spectra < threshold] = threshold
    sum_model_spectra = torch.sum(torch.where(nan_mask,zero_sub,model_spectra),axis=1)
    sum_model_spectra = torch.unsqueeze(sum_model_spectra,axis=1)
    model_spectra = torch.div(model_spectra,sum_model_spectra)
    # calculate loss value
    if not isinstance(target_spectra,torch.Tensor):
        target_spectra = torch.tensor(target_spectra)
    target_spectra = target_spectra.to(torch_device)
    loss = torch.ones_like(target_spectra)
    loss = loss.to(torch_device)
    target_spectra[nan_mask]=1
    model_spectra[nan_mask]=1
    loss = torch.mean((model_spectra-target_spectra)**2,dim=1)
    return loss

def wasserstein(model_spectra: torch.tensor, target_spectra: torch.tensor, threshold: float = 1e-8, eps: float = 1e-8, torch_device: str = 'cpu') -> torch.tensor:
    # normalize the model spectra before comparison
    nan_mask=torch.isnan(target_spectra)+torch.isnan(model_spectra)
    nan_mask=nan_mask.to(device=torch_device)
    zero_sub=torch.zeros_like(target_spectra,device=torch_device)
    model_spectra = model_spectra.to(torch_device)
    model_spectra[model_spectra < threshold] = threshold
    sum_model_spectra = torch.sum(torch.where(nan_mask,zero_sub,model_spectra),axis=1)
    sum_model_spectra = torch.unsqueeze(sum_model_spectra,axis=1)
    model_spectra = torch.div(model_spectra,sum_model_spectra)
    # cumulative spectra
    if not isinstance(target_spectra,torch.Tensor):
        target_spectra = torch.tensor(target_spectra)
    target_spectra = target_spectra.to(torch_device)
    target_spectra[nan_mask]=0
    model_spectra[nan_mask]=0
    cum_model = torch.ones_like(model_spectra)
    cum_model = cum_model.to(torch_device)
    cum_targets = torch.ones_like(target_spectra)
    cum_targets = cum_targets.to(torch_device)
    cum_model = torch.cumsum(model_spectra,dim=1)
    cum_targets = torch.cumsum(target_spectra,dim=1)
    # calculate loss
    loss = torch.ones_like(target_spectra)
    loss = loss.to(torch_device)
    loss = torch.add(cum_model,torch.mul(cum_targets,-1))
    loss = torch.abs(loss)
    loss = torch.sum(loss,axis=1)
    return loss

def pre_normalize_targets(targets: List[List[float]], threshold: float = 1e-8, torch_device: str = 'cpu', batch_size: int = 50) -> List[List[float]]:
    normalized_targets = []

    num_iters, iter_step = len(targets), batch_size

    for i in trange(0, num_iters, iter_step):
        with torch.no_grad():
            # Prepare batch
            batch = targets[i:i + iter_step]
            batch = torch.tensor(batch,dtype=float,device=torch_device)
            batch[batch < threshold] = threshold
            nan_mask=torch.isnan(batch)
            batch[nan_mask]=0
            batch_sums = torch.sum(batch,axis=1)
            batch_sums = torch.unsqueeze(batch_sums,axis=1)
            norm_batch = torch.div(batch,batch_sums)
            norm_batch[nan_mask] = float('nan')
            norm_batch = norm_batch.data.cpu().tolist()

            # Collect vectors
            normalized_targets.extend(norm_batch)
    return normalized_targets

def generate_conv_matrix(length: int = 1801, stdev: float = 0.0, torch_device: str = 'cpu') -> torch.tensor:
    conv_matrix = torch.eye(length,dtype=float,device=torch_device)
    # conv_matrix = torch.unsqueeze(conv_matrix,dim=0)
    if stdev != 0:
        conv_vector=[0]*length
        for vector_i in range(length):
            conv_vector[vector_i]=(1/np.sqrt(2*np.pi*np.square(stdev)))*np.exp(-1*(np.square(vector_i)/(2*np.square(stdev))))
        for source_i in range(length):
            for conv_i in range(length):
                # conv_matrix[0,source_i,conv_i]=conv_vector[abs(source_i-conv_i)]
                conv_matrix[source_i,conv_i]=conv_vector[abs(source_i-conv_i)]
    return conv_matrix


def roundrobin_sid(spectra: torch.Tensor, threshold: float = 1e-8, torch_device: str = 'cpu', stdev: float = 0.0) -> torch.Tensor:
    """
    Takes a block of input spectra and makes a pairwise comparison between each of the input spectra for a given molecule,
    returning a list of the spectral informations divergences. Also saves a file with the list and reference to which sid
    came from which pair of spectra. To be used evaluating the variation between an ensemble of model spectrum predictions.

    :spectra: A 3D tensor containing each of the spectra to be compared. Different molecules along axis=0,
    different ensemble spectra along axis=1, different frequency bins along axis=2.
    :threshold: SID calculation requires positive values in each position, this value is used to replace any zero or negative values.
    :torch_device: Tag for pytorch device to use for calculation. If run in chemprop, this will be args.device.
    :save_file: A location to save the sid results for each pair, does not write unless specified.
    :stdev: If the spectra are to be gaussian convolved before sid comparison, they will be spread using a gaussian of this standard deviation,
    defined in terms of the number of target bins not the units of the bin labels.
    :return: A tensor containing a list of SIDs for each pairwise combination of spectra along axis=1, for each molecule provided along axis=0.
    """
    spectra=spectra.to(device = torch_device,dtype=float)

    if stdev != 0.0:
        conv_matrix = generate_conv_matrix(length=len(spectra[0,0]),stdev=stdev,torch_device=torch_device)

    ensemble_size=spectra.size()[1]
    spectrum_size=spectra.size()[2]
    number_pairs=sum(range(ensemble_size))

    ensemble_sids=torch.zeros([0,number_pairs],device=torch_device,dtype=float) #0*n

    for i in range(len(spectra)):
        with torch.no_grad():
            mol_spectra = spectra[i] #10*1801
            nan_mask=torch.isnan(mol_spectra[0]) #1801
            nan_mask=nan_mask.to(device=torch_device)
            mol_spectra = mol_spectra.to(device=torch_device,dtype=float)
            mol_spectra[mol_spectra<threshold] = threshold
            mol_spectra[:,nan_mask]=0
            if stdev != 0.0:
                mol_spectra = torch.matmul(mol_spectra,conv_matrix)
            mol_sums = torch.sum(mol_spectra, axis=1) #10
            mol_sums = torch.unsqueeze(mol_sums,axis=1) #10*1
            mol_norm = torch.div(mol_spectra,mol_sums) #10*1801
            mol_norm[:,nan_mask]=1
            ensemble_head = torch.zeros([0,spectrum_size],device=torch_device,dtype=float) #0*1801
            ensemble_tail = torch.zeros([0,spectrum_size],device=torch_device,dtype=float) #0*1801
            for j in range(len(mol_norm)-1):
                ensemble_tail = torch.cat((ensemble_tail,mol_norm[j+1:]),axis=0) #n*1801
                ensemble_head = torch.cat((ensemble_head,mol_norm[:-j-1]),axis=0) #n*1801
            mol_loss = torch.zeros_like(ensemble_head,device=torch_device,dtype=float) #n*1801
            mol_loss = torch.mul(torch.log(torch.div(ensemble_head,ensemble_tail)),ensemble_head) \
                + torch.mul(torch.log(torch.div(ensemble_tail,ensemble_head)),ensemble_tail)
            mol_loss[:,nan_mask]=0
            mol_loss = torch.sum(mol_loss,axis=1) #n
            mol_loss = torch.unsqueeze(mol_loss,axis=0)
            ensemble_sids = torch.cat((ensemble_sids,mol_loss),axis=0) #0*n
    return ensemble_sids

def apply_spectral_mask(spectral_mask: List[List[float]],spectra: List[List[float]],features: List[List[float]], torch_device: str = 'cpu', batch_size: int = 50):

    masked_spectra = []
    spectral_mask=np.array(spectral_mask,dtype=float)
    spectral_mask=torch.from_numpy(spectral_mask).to(device=torch_device)

    num_iters, iter_step = len(spectra), batch_size

    for i in trange(0, num_iters, iter_step):
        with torch.no_grad():
            # Prepare batch
            batch_spectra = spectra[i:i + iter_step]
            batch_features = features[i:i + iter_step]
            batch_spectra = torch.tensor(batch_spectra,dtype=float,device=torch_device)
            batch_features = torch.tensor(batch_features,dtype=float,device=torch_device)

            # Extract phase features from batch_features
            phase_features = batch_features[:,-len(spectral_mask):]
            batch_mask=torch.matmul(phase_features,spectral_mask).bool()

            batch_spectra[~batch_mask]=float('nan')

            batch_spectra = batch_spectra.data.cpu().tolist()

            # Collect vectors
            masked_spectra.extend(batch_spectra)

    return masked_spectra
