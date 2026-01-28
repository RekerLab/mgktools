#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model interpretability module.

This module provides classes and functions for understanding model predictions
through atomic and molecular attribution, helping to identify which structural
features contribute most to predicted properties.

Key Functions
-------------
interpret_training_mols
    Molecular-level interpretation via training set contribution decomposition.
interpret_atoms
    Atomic-level interpretation showing atom contributions to predictions.
get_node_graphs
    Create atom-masked graphs for atomic attribution.
"""
