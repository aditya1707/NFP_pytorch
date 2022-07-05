import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def _one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: int(x == s), allowable_set))

def _get_atom_features(mol):
    '''
    Function to get atom features from RDKit

    Parameters
    ----------
    mol: rdkit molecule
        object containing rdkit molecule
    
    Returns
    -------
    atom_features: torch.tensor
        tensor containing atom features
    '''
    atomic_number = []
    num_hs = []
    atom_degree = []
    implicit_valence = []
    hybridization =[]
    isaromatic = []

    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))
        atom_degree.append(atom.GetDegree())
        implicit_valence.append(atom.GetImplicitValence())
        hybridization.append(atom.GetHybridization())
        isaromatic.append(atom.GetIsAromatic())
    # return torch.tensor([atomic_number, num_hs]).t()
    atom_features = torch.tensor([atomic_number, num_hs,atom_degree,implicit_valence, hybridization, isaromatic]).t()
    return atom_features

def _get_edge_index(mol):
    '''
    '''
    row, col = [], []
    
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
    return torch.tensor([row, col], dtype=torch.long)

def prepare_dataloader(mol_list,batch_size):
    '''
    '''
    data_list = []
    for i, mol in enumerate(mol_list):

        x = _get_atom_features(mol)
        print(x)
        edge_index = _get_edge_index(mol)

        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

    dl = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    return dl, data_list