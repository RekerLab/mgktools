#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data handling module for molecular datasets.

This module provides classes and utilities for managing molecular datasets,
including caching of molecular graphs and features, data point representation,
dataset operations, and train/validation/test splitting.

Key Classes
-----------
CachedDict
    Cache for SMILES-to-graph and SMILES-to-features conversions.
Datapoint
    Container for a single molecular data point.
Dataset
    Main dataset container for molecular property prediction.

Key Functions
-------------
get_data_from_index
    Extract subset of dataset by indices.
dataset_split
    Split dataset into train/validation/test sets.
"""
