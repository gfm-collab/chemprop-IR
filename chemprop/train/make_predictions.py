from argparse import Namespace
import csv
from typing import List, Optional
import os

import numpy as np
import torch
from tqdm import tqdm

from .predict import predict
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers
from chemprop.train.spectral_loss import roundrobin_sid

def make_predictions(args: Namespace, smiles: List[str] = None) -> List[Optional[List[float]]]:
    """
    Makes predictions. If smiles is provided, makes predictions on smiles. Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :param smiles: Smiles to make predictions on.
    :return: A list of lists of target predictions.
    """
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    print('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])

    # Update args with training arguments
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print('Loading data')
    if smiles is not None:
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False, args=args)
    else:
        test_data = get_data(path=args.test_path, args=args, use_compound_names=args.use_compound_names, skip_invalid_smiles=False)

    print('Validating SMILES')
    valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
    full_data = test_data
    test_data = MoleculeDataset([test_data[i] for i in valid_indices])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    if args.use_compound_names:
        compound_names = test_data.compound_names()
    print(f'Test size = {len(test_data):,}')

    # Normalize features
    if train_args.features_scaling:
        test_data.normalize_features(features_scaler)

    # Predict with each model individually and sum predictions
    if args.dataset_type == 'multiclass':
        sum_preds = np.zeros((len(test_data), args.num_tasks, args.multiclass_num_classes))
        sum_epi_uncs = np.zeros((len(test_data), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_preds = np.zeros((len(test_data), args.num_tasks))
        sum_epi_uncs = np.zeros((len(test_data), args.num_tasks))

    all_preds = np.zeros((len(test_data), args.num_tasks, len(args.checkpoint_paths)))

    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for index, checkpoint_path in enumerate(tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths))):
        # Load model
        model = load_checkpoint(checkpoint_path, cuda=args.cuda)
        model_preds = predict(
            model=model,
            args=args,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )
        sum_preds += np.array(model_preds)
        if args.ensemble_variance:
            all_preds[:, :, index] = model_preds

    # Ensemble predictions
    if args.ensemble_variance:
        # Use ensemble variance to estimate uncertainty. This overwrites existing uncertainty estimates, if any
        avg_preds = sum_preds / len(args.checkpoint_paths)
        avg_preds = avg_preds.tolist()

        # sid_loss -> ensemble_sid for spectra "variance"
        if args.dataset_type == 'spectra':
            # ensemble_sid takes a tensor < molecules x spectra x frequencies >
            _roundrobin_sid_input = np.transpose(all_preds, (0,2,1))
            epi_uncs_tensor = roundrobin_sid(torch.tensor(_roundrobin_sid_input, dtype=torch.float32, device=args.device),threshold=args.sm_thresh, torch_device=args.device,stdev=args.ensemble_variance_conv)

            # epistemic uncertainty of a molecule: mean of the convolved ensemble sid versus average output results for that molecule
            epi_uncs = epi_uncs_tensor.mean(axis=1).tolist()

        # standard variance
        else:
            epi_uncs = np.var(all_preds, axis=2)
            epi_uncs = epi_uncs.tolist()

    else:
        # Use another method to estimate uncertainty.
        # preds <- mean(preds), ale_uncs <- mean(ale_uncs), epi_uncs <- mean(epi_uncs)
        avg_preds = sum_preds / len(args.checkpoint_paths)
        avg_preds = avg_preds.tolist()

        epi_uncs = sum_epi_uncs / len(args.checkpoint_paths)
        epi_uncs = epi_uncs.tolist()

    # Save predictions
    assert len(test_data) == len(avg_preds)
    assert len(test_data) == len(epi_uncs)
    print(f'Saving predictions to {args.preds_path}')

    # Put Nones for invalid smiles
    full_preds = [None] * len(full_data)
    full_epi_uncs = [None] * len(full_data)
    for i, si in enumerate(valid_indices):
        full_preds[si] = avg_preds[i]
        full_epi_uncs[si] = epi_uncs[i]
    avg_preds = full_preds
    epi_uncs = full_epi_uncs
    test_smiles = full_data.smiles()

    # Write predictions
    with open(args.preds_path, 'w') as f:
        writer = csv.writer(f)

        header = []

        if args.use_compound_names:
            header.append('compound_names')

        header.append('smiles')

        if args.dataset_type == 'multiclass':
            for name in args.task_names:
                for i in range(args.multiclass_num_classes):
                    header.append(name + '_class' + str(i))
        else:
            header.extend(args.task_names)
            if args.ensemble_variance:
                if args.dataset_type=='spectra':
                    header.append('epi_unc')
                else:
                    header.extend([tn + "_epi_unc" for tn in args.task_names])
        writer.writerow(header)

        for i in range(len(avg_preds)):
            row = []

            if args.use_compound_names:
                row.append(compound_names[i])

            row.append(test_smiles[i])

            if avg_preds[i] is not None:
                if args.dataset_type == 'multiclass':
                    for task_probs in avg_preds[i]:
                        row.extend(task_probs)
                else:
                    row.extend(avg_preds[i])
                    if args.ensemble_variance:
                        if args.dataset_type=='spectra':
                            row.append(epi_uncs[i])
                        else:
                            row.extend(epi_uncs[i])
            else:
                if args.dataset_type == 'multiclass':
                    row.extend([''] * args.num_tasks * args.multiclass_num_classes)
                else:
                    row.extend([''] * args.num_tasks)
                    if args.ensemble_variance:
                        if args.dataset_type=='spectra':
                            row.append('')
                        else:
                            row.extend([''] * args.num_tasks)

            writer.writerow(row)

    return avg_preds
