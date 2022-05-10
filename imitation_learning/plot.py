#%%
import tensorboard as tb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
sns.set_context("talk")
sns.set_style("ticks")

#%%
logging_dir="./tensorboard"
event_paths=glob.glob(os.path.join(logging_dir, "*","event*"))
event_paths

#%%
import tensorflow as tf

#%%
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)

#%%
for path in event_paths:
    if 'test' not in path:
        for e in my_summary_iterator(path):
            for v in e.summary.value:
                print(v)

# %%
