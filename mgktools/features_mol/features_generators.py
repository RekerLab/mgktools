#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Molecular feature generators for fingerprints and descriptors.

This module provides the FeaturesGenerator class for computing various types
of molecular fingerprints and descriptors from SMILES strings or RDKit molecules.

Classes
-------
FeaturesGenerator
    Generator for molecular fingerprints and descriptors.

Constants
---------
AVAILABLE_FEATURES_GENERATORS : list
    List of supported feature generator names.
"""
from typing import Union
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Avalon.pyAvalonTools import GetAvalonFP, GetAvalonCountFP
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors


AVAILABLE_FEATURES_GENERATORS = [
    'morgan', 'morgan_count', 'rdkit_208', 'rdkit_2d', 'rdkit_2d_normalized',
    'rdkit_topol', 'layered', 'torsion', 'atom_pair', 'avalon', 'avalon_count',
    'maccskey', 'pattern'
]


class FeaturesGenerator:
    """
    Generator for molecular fingerprints and descriptors.

    This class supports multiple types of molecular representations including
    various fingerprints (Morgan, MACCS, Avalon, etc.) and molecular descriptors
    (RDKit 2D descriptors).

    Parameters
    ----------
    features_generator_name : str
        Name of the feature generator. Supported options:
        - 'morgan': Binary Morgan fingerprint
        - 'morgan_count': Count-based Morgan fingerprint
        - 'rdkit_208': RDKit 208 molecular descriptors
        - 'rdkit_2d': RDKit 2D descriptors
        - 'rdkit_2d_normalized': Normalized RDKit 2D descriptors
        - 'rdkit_topol': RDKit topological fingerprint
        - 'layered': Layered fingerprint
        - 'torsion': Topological torsion fingerprint
        - 'atom_pair': Atom pair fingerprint
        - 'avalon': Avalon fingerprint
        - 'avalon_count': Count-based Avalon fingerprint
        - 'maccskey': MACCS keys fingerprint
        - 'pattern': Pattern fingerprint
    radius : int, optional
        Radius for Morgan fingerprint, by default 2.
    num_bits : int, optional
        Number of bits for fingerprints, by default 2048.
    atomInvariantsGenerator : bool, optional
        Whether to use atom invariants for Morgan fingerprint, by default False.

    Attributes
    ----------
    features_generator_name : str
        Name of the configured feature generator.
    radius : int
        Morgan fingerprint radius.
    num_bits : int
        Number of fingerprint bits.
    generator : object
        Internal generator object (for rdkit_2d types).

    Examples
    --------
    >>> fg = FeaturesGenerator('morgan', radius=2, num_bits=2048)
    >>> features = fg('CCO')  # Ethanol
    >>> len(features)
    2048

    >>> fg = FeaturesGenerator('rdkit_2d_normalized')
    >>> features = fg('c1ccccc1')  # Benzene

    Notes
    -----
    Some generators (rdkit_2d, rdkit_2d_normalized) use Boost.Python objects
    that cannot be pickled. Use n_jobs=1 when parallelizing with these generators.
    """

    def __init__(self, features_generator_name: str,
                 radius: int = 2,
                 num_bits: int = 2048,
                 atomInvariantsGenerator: bool = False):
        """Initialize feature generator with specified parameters."""
        self.features_generator_name = features_generator_name
        self.radius = radius
        self.num_bits = num_bits
        self.atomInvariantsGenerator = atomInvariantsGenerator
        # Lazy-initialized generators (not created until first use)
        # This allows the object to be pickled for parallel processing
        self._morgan_generator = None
        self._torsion_generator = None
        self._atom_pair_generator = None
        self._rdkit_fp_generator = None
        self._rdkit_2d_generator = None

    def __getstate__(self):
        """Return state for pickling, excluding non-picklable generators."""
        return {
            'features_generator_name': self.features_generator_name,
            'radius': self.radius,
            'num_bits': self.num_bits,
            'atomInvariantsGenerator': self.atomInvariantsGenerator,
        }

    def __setstate__(self, state):
        """Restore state from pickle, reinitializing generators as None."""
        self.features_generator_name = state['features_generator_name']
        self.radius = state['radius']
        self.num_bits = state['num_bits']
        self.atomInvariantsGenerator = state['atomInvariantsGenerator']
        # Reset generators to None - they will be lazily recreated
        self._morgan_generator = None
        self._torsion_generator = None
        self._atom_pair_generator = None
        self._rdkit_fp_generator = None
        self._rdkit_2d_generator = None

    def _get_morgan_generator(self):
        """Lazy initialization of Morgan generator."""
        if self._morgan_generator is None:
            if self.atomInvariantsGenerator:
                invgen = AllChem.GetMorganFeatureAtomInvGen()
                self._morgan_generator = AllChem.GetMorganGenerator(
                    radius=self.radius, fpSize=self.num_bits, atomInvariantsGenerator=invgen)
            else:
                self._morgan_generator = AllChem.GetMorganGenerator(
                    radius=self.radius, fpSize=self.num_bits)
        return self._morgan_generator

    def _get_torsion_generator(self):
        """Lazy initialization of torsion generator."""
        if self._torsion_generator is None:
            self._torsion_generator = AllChem.GetTopologicalTorsionGenerator(fpSize=self.num_bits)
        return self._torsion_generator

    def _get_atom_pair_generator(self):
        """Lazy initialization of atom pair generator."""
        if self._atom_pair_generator is None:
            self._atom_pair_generator = AllChem.GetAtomPairGenerator(fpSize=self.num_bits)
        return self._atom_pair_generator

    def _get_rdkit_fp_generator(self):
        """Lazy initialization of RDKit fingerprint generator."""
        if self._rdkit_fp_generator is None:
            self._rdkit_fp_generator = AllChem.GetRDKitFPGenerator(fpSize=self.num_bits)
        return self._rdkit_fp_generator

    def _get_rdkit_2d_generator(self):
        """Lazy initialization of RDKit 2D descriptor generator."""
        if self._rdkit_2d_generator is None:
            if self.features_generator_name == 'rdkit_2d':
                self._rdkit_2d_generator = rdDescriptors.RDKit2D()
            elif self.features_generator_name == 'rdkit_2d_normalized':
                self._rdkit_2d_generator = rdNormalizedDescriptors.RDKit2DNormalized()
        return self._rdkit_2d_generator

    def __call__(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generate features for a molecule.

        Parameters
        ----------
        mol : str or Chem.Mol
            Input molecule as SMILES string or RDKit Mol object.

        Returns
        -------
        np.ndarray
            1D array of molecular features.

        Raises
        ------
        ValueError
            If features_generator_name is not supported.
        """
        if self.features_generator_name == 'morgan':
            return self.morgan_binary_features_generator(mol)
        elif self.features_generator_name == 'morgan_count':
            return self.morgan_counts_features_generator(mol)
        elif self.features_generator_name == 'rdkit_2d':
            return self.rdkit_2d_features_generator(mol)
        elif self.features_generator_name == 'rdkit_2d_normalized':
            return self.rdkit_2d_normalized_features_generator(mol)
        elif self.features_generator_name == 'rdkit_208':
            return self.rdkit_208_features_generator(mol)
        elif self.features_generator_name == 'rdkit_topol':
            return self.rdkit_topological_features_generator(mol)
        elif self.features_generator_name == 'layered':
            return self.layered_features_generator(mol)
        elif self.features_generator_name == 'torsion':
            return self.torsion_features_generator(mol)
        elif self.features_generator_name == 'atom_pair':
            return self.atom_pair_features_generator(mol)
        elif self.features_generator_name == 'avalon':
            return self.avalon_features_generator(mol)
        elif self.features_generator_name == 'avalon_count':
            return self.avalon_count_features_generator(mol)
        elif self.features_generator_name == 'maccskey':
            return self.maccskey_features_generator(mol)
        elif self.features_generator_name == 'pattern':
            return self.pattern_features_generator(mol)
        else:
            raise ValueError(f'unknown features generator: {self.features_generator_name}')

    def morgan_binary_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generate binary Morgan fingerprint for a molecule.

        Parameters
        ----------
        mol : str or Chem.Mol
            Input molecule as SMILES string or RDKit Mol object.

        Returns
        -------
        np.ndarray
            1D binary array of Morgan fingerprint bits.
        """
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(self._get_morgan_generator().GetFingerprint(mol).ToList())

    def morgan_counts_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generate count-based Morgan fingerprint for a molecule.

        Parameters
        ----------
        mol : str or Chem.Mol
            Input molecule as SMILES string or RDKit Mol object.

        Returns
        -------
        np.ndarray
            1D array of Morgan fingerprint counts.
        """
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(self._get_morgan_generator().GetCountFingerprint(mol).ToList())

    def rdkit_2d_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generate RDKit 2D descriptors for a molecule.

        Parameters
        ----------
        mol : str or Chem.Mol
            Input molecule as SMILES string or RDKit Mol object.

        Returns
        -------
        np.ndarray
            1D array of RDKit 2D descriptor values.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if isinstance(mol, Chem.Mol) else mol
        return np.array(self._get_rdkit_2d_generator().process(smiles)[1:])

    def rdkit_2d_normalized_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generate normalized RDKit 2D descriptors for a molecule.

        Parameters
        ----------
        mol : str or Chem.Mol
            Input molecule as SMILES string or RDKit Mol object.

        Returns
        -------
        np.ndarray
            1D array of normalized RDKit 2D descriptor values.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if isinstance(mol, Chem.Mol) else mol
        return np.array(self._get_rdkit_2d_generator().process(smiles)[1:])

    @staticmethod
    def rdkit_208_features_generator(mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generate RDKit 208 molecular descriptors for a molecule.

        Parameters
        ----------
        mol : str or Chem.Mol
            Input molecule as SMILES string or RDKit Mol object.

        Returns
        -------
        np.ndarray
            1D array of 208+ molecular descriptor values.

        Notes
        -----
        Large values (> float32 max) are capped, and NaN values are set to 0.
        """
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        descr = Descriptors._descList
        calc = [x[1] for x in descr]
        ds_n = []
        for d in calc:
            v = d(mol)
            if v > np.finfo(np.float32).max:
                ds_n.append(np.finfo(np.float32).max)
            elif math.isnan(v):
                ds_n.append(np.float32(0.0))
            else:
                ds_n.append(np.float32(v))
        return np.array(ds_n)

    def layered_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generate layered fingerprint for a molecule.

        Parameters
        ----------
        mol : str or Chem.Mol
            Input molecule as SMILES string or RDKit Mol object.

        Returns
        -------
        np.ndarray
            1D binary array of layered fingerprint bits.
        """
        return np.array(Chem.LayeredFingerprint(mol, fpSize=self.num_bits).ToList())

    def torsion_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generate topological torsion fingerprint for a molecule.

        Parameters
        ----------
        mol : str or Chem.Mol
            Input molecule as SMILES string or RDKit Mol object.

        Returns
        -------
        np.ndarray
            1D array of topological torsion fingerprint.
        """
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(self._get_torsion_generator().GetFingerprint(mol).ToList())

    def atom_pair_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generate atom pair fingerprint for a molecule.

        Parameters
        ----------
        mol : str or Chem.Mol
            Input molecule as SMILES string or RDKit Mol object.

        Returns
        -------
        np.ndarray
            1D array of atom pair fingerprint.
        """
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(self._get_atom_pair_generator().GetFingerprint(mol).ToList())

    def avalon_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generate Avalon fingerprint for a molecule.

        Parameters
        ----------
        mol : str or Chem.Mol
            Input molecule as SMILES string or RDKit Mol object.

        Returns
        -------
        np.ndarray
            1D binary array of Avalon fingerprint bits.
        """
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(GetAvalonFP(mol, nBits=self.num_bits).ToList())

    def avalon_count_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generate count-based Avalon fingerprint for a molecule.

        Parameters
        ----------
        mol : str or Chem.Mol
            Input molecule as SMILES string or RDKit Mol object.

        Returns
        -------
        np.ndarray
            1D array of Avalon fingerprint counts.
        """
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(GetAvalonCountFP(mol, nBits=self.num_bits).ToList())

    def maccskey_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generate MACCS keys fingerprint for a molecule.

        Parameters
        ----------
        mol : str or Chem.Mol
            Input molecule as SMILES string or RDKit Mol object.

        Returns
        -------
        np.ndarray
            1D binary array of 167 MACCS key bits.
        """
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(AllChem.GetMACCSKeysFingerprint(mol).ToList())

    def pattern_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generate pattern fingerprint for a molecule.

        Parameters
        ----------
        mol : str or Chem.Mol
            Input molecule as SMILES string or RDKit Mol object.

        Returns
        -------
        np.ndarray
            1D binary array of pattern fingerprint bits.
        """
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(AllChem.rdmolops.PatternFingerprint(mol, fpSize=self.num_bits).ToList())

    def rdkit_topological_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generate RDKit topological fingerprint for a molecule.

        Parameters
        ----------
        mol : str or Chem.Mol
            Input molecule as SMILES string or RDKit Mol object.

        Returns
        -------
        np.ndarray
            1D binary array of RDKit topological fingerprint bits.
        """
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(self._get_rdkit_fp_generator().GetFingerprint(mol).ToList())
