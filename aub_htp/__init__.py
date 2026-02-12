# aub_htp/__init__.py
"""
AUB Heavy-Tails Package (AUB-HTP)

A comprehensive toolkit for working with heavy-tailed distributions,
including PDF generation and machine learning methods.
"""
# PDF generation
from .pdf import generate_alpha_stable_pdf

from ._alpha_stable import alpha_stable_gen

alpha_stable = alpha_stable_gen("alpha_stable")