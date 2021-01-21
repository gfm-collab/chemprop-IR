from argparse import Namespace
import csv
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .predict import predict
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers


def create_fingerprints(args: Namespace, smiles: List[str] = None) -> List[Optional[List[float]]]:
    """
    Create fingerprint vectors for the specified molecules. If smiles is provided, makes predictions on smiles. Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :param smiles: Smiles to make predictions on.
    :return: A list of fingerprint vectors (list of floats)
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

    print(f'Encoding smiles into a fingerprint vector from a single model')
    # Load model
    model = load_checkpoint(args.checkpoint_paths[0], current_args=args, cuda=args.cuda)
    if hasattr(model,'spectral_mask'):
        delattr(model,'spectral_mask')
    model_preds = predict(
        model=model,
        args=args,
        data=test_data,
        batch_size=args.batch_size,
    )

    # Save predictions
    assert len(test_data) == len(model_preds)
    print(f'Saving predictions to {args.preds_path}')

    # Put Nones for invalid smiles
    full_preds = [None] * len(full_data)
    for i, si in enumerate(valid_indices):
        full_preds[si] = model_preds[i]
    model_preds = full_preds
    test_smiles = full_data.smiles()

    # Write predictions
    with open(args.preds_path, 'w') as f:
        writer = csv.writer(f)

        header = []

        if args.use_compound_names:
            header.append('compound_names')

        header.extend(['smiles'])
        header.extend(['fp{}'.format(x) for x in range(1,args.hidden_size+1)])

        writer.writerow(header)

        for i in range(len(model_preds)):
            row = []

            if args.use_compound_names:
                row.append(compound_names[i])

            row.append(test_smiles[i])

            if model_preds[i] is not None:
                row.extend(model_preds[i][:args.hidden_size])
            else:
                row.extend([''] * args.hidden_size)

            writer.writerow(row)

    return model_preds
