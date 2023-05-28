
from adjustText import adjust_text

import codecs

from collections import namedtuple

import csv

import datetime

import glob

from importlib import reload

from itertools import product, combinations_with_replacement

import math

import matplotlib.pyplot as plt

from nbconvert import HTMLExporter
import nbformat

from numpy import amin, amax, arctanh, array, array_str, arange, argmax, argmin, argsort, column_stack
from numpy import c_, concatenate, copy, corrcoef, delete
from numpy import diag, digitize, divide, dot, empty, exp, float64, fromiter, isinf, isnan
from numpy import Inf, linspace, loadtxt, log, logical_and, maximum, mean, median, nan, newaxis, minimum, ndarray, newaxis, ones
from numpy import poly1d, polyfit, percentile
from numpy import quantile, repeat, roll, round, row_stack, savetxt, setdiff1d, sort, sqrt, std, sum, tanh, transpose, unique, where, zeros
from numpy.linalg import inv, lstsq, matrix_rank, solve, svd
from numpy.random import choice, rand, randint, permutation, seed
from numpy import append as np_append
from numpy import delete as np_delete
from numpy import round as np_round

import os
                                                               
import pandas as pd

import pickle

import random

import seaborn as sns

from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit
from scipy.stats import t, pearsonr, skew, kurtosis
from scipy.special import expit, logit
from scipy import linalg 

from sklearn import datasets, linear_model
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler   
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor, export_text

import statsmodels.api as sm

import sys

from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import xgboost as xgb
