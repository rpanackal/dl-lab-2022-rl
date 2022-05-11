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
training_log = pd.DataFrame(columns=['history_length', 'learning_rate', 'step','training_loss', 
'training_accuracy', 'validation_loss', 'validation_accuracy'])
for path in event_paths:
    row=[]
    tmp=[]
    if 'test' not in path:
        history_length=5
        if "h1" in path:
            history_length=1
        elif "h3" in path:
            history_length=3        
        lr=0.00001
        if "0.0001" in path:
            lr=0.0001
        elif "0.001" in path:
            lr=0.001
        elif "0.01" in path:
            lr=0.01
        
        for e in my_summary_iterator(path):            
            if e.step==0:
                continue
            for v in e.summary.value:                
                if v.tag=="training_loss":
                    tmp=[]
                tmp.append(v.simple_value)                
            if len(tmp)==4:
                row.append([history_length, lr, e.step, 
                tmp[0], tmp[1], tmp[2], tmp[3]])
                tmp_df = pd.Series(row[0], index=['history_length', 'learning_rate', 
                'step','training_loss', 'training_accuracy', 'validation_loss', 
                'validation_accuracy'])
                row=[]             
                training_log=training_log.append(tmp_df, ignore_index=True)
    
training_log   
training_log.to_csv('training_log.csv')


# %%
test_log = pd.DataFrame(columns=['episode_id', 
'episode_return', 'history_length', 'learning_rate', 'step'])
for path in event_paths:
    row=[]
    tmp=[]
    if 'test' in path and 'lr' in path:
        history_length=5
        if "h1" in path:
            history_length=1
        elif "h3" in path:
            history_length=3        
        lr=0.00001
        if "0.0001" in path:
            lr=0.0001
        elif "0.001" in path:
            lr=0.001
        elif "0.01" in path:
            lr=0.01
        
        for e in my_summary_iterator(path):            
            if e.step==0:
                continue
            for v in e.summary.value:                
                if v.tag=="episode_id":
                    tmp=[]
                tmp.append(v.simple_value)                
            if len(tmp)==4:
                row.append([tmp[0], tmp[1], history_length, lr, e.step])
                tmp_df = pd.Series(row[0], index=['episode_id', 
                'episode_return', 'history_length', 'learning_rate', 'step'])
                row=[]             
                test_log=test_log.append(tmp_df, ignore_index=True)
    
test_log   
test_log.to_csv('test_log.csv')


#%%
#training_log['mov_avg'] = df_nat['new_cases'].rolling(7).sum()
#training_log2= training_log.rolling(window=7, method="table", engine='numba')
training_log["new_t_l"]=training_log["training_loss"].rolling(window=6).mean()
training_log["new_t_a"]=training_log["training_accuracy"].rolling(window=6).mean()
training_log["new_v_l"]=training_log["validation_loss"].rolling(window=6).mean()
training_log["new_v_a"]=training_log["validation_accuracy"].rolling(window=6).mean()
#%%
sns.set_style("whitegrid")
ERROR_LABELS = {"new_t_l": "Training Loss", 
"new_t_a":"Training Accuracy", 
"new_v_l":"Validation Loss", 
'new_v_a':"Validation Accuracy"}
fig, axes = plt.subplots(2, 2, figsize=(15.5, 15.5))
for (error, label), ax in zip(ERROR_LABELS.items(), axes.flatten()):
    sns.lineplot(data=training_log, x="step", y=error, hue="history_length", style="learning_rate", ax=ax, size_norm=True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend('', frameon=False)
    ax.set_title(label)
    if "Accuracy" in label:
        ax.set_ylabel(label + " (%)" )
    else:
        ax.set_ylabel(label)
    ax.set_xlabel("Num. of Steps")
    if error == "new_t_l":
        fig.legend(handles, labels, bbox_to_anchor=(0.3, 1.2), loc=2, borderaxespad=0., ncol=2)
    fig.tight_layout()
    plt.savefig("training2.png", bbox_inches='tight')


#%%
sns.set_style("whitegrid")
sns.lineplot(data=test_log, x="step", y="episode_return", hue="history_length", style="learning_rate")
plt.show()
plt.savefig("test.png", bbox_inches='tight')



#%%
test_log["new_reward"]=test_log["episode_return"].rolling(window=6).mean()
sns.set_style("whitegrid")
ERROR_LABELS = {"new_reward": "Score"}
fig, axes = plt.subplots(1, 1, figsize=(15.5, 15.5))
for (error, label) in ERROR_LABELS.items():
    sns.lineplot(data=test_log, x="step", y=error, hue="history_length", style="learning_rate", ax=axes)
    handles, labels = axes.get_legend_handles_labels()
    #ax.legend('', frameon=False)
    axes.set_title(label)
    #if "Accuracy" in label:
    #    ax.set_ylabel(error + " (%)" )
    #else:
    axes.set_ylabel(label)
    axes.set_xlabel("Num. of Steps")
    #if error == "training_loss":
    #fig.legend(handles, labels, bbox_to_anchor=(0.3, 1.2), loc=2, borderaxespad=0., ncol=2)
    fig.tight_layout()
    plt.savefig("test2.png", bbox_inches='tight')
# %%
a=test_log.sort_values(by="episode_return")
a.to_csv('best.csv')
# %%
