#!/usr/bin/python

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np

def iqr(x):
"""
Calculates interquartile range
"""
    return np.subtract(*np.percentile(x, [75, 25]))

raw_file = pd.read_csv("data.csv",
                quotechar='"',encoding='UTF-8')

"""
Bar chart of label counts
"""
grouped = raw_file.groupby(['label']).size().reset_index()
n_groups = 1
fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
rects1 = plt.bar(index, grouped[grouped['label']==-1][0], 
                 bar_width,
                 alpha=opacity,
                 color='b',
                 label='Non Coreference')
rects2 = plt.bar(index + bar_width, 
                 grouped[grouped['label']==1][0], 
                 bar_width,
                 alpha=opacity,
                 color='r',
                 label='Coreference')
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Count of Labels')
plt.xticks(index + bar_width)
plt.legend()

plt.tight_layout()
pylab.savefig('LabelDistribution.png', bbox_inches='tight')
plt.close()



"""
Bar charts of label counts by each field
"""
fields = list(set(raw_file.columns.values) - set(['label']))
for field in fields:
    n = len(raw_file[field])
    grouped = raw_file.groupby([field,'label']).size().reset_index()
    no_coref = grouped[grouped['label']==-1]
    coref = grouped[grouped['label']==1]
    if np.dtype(grouped.dtypes[field])=='O':
    # If datatype is a numpy object (categorical field)
        distinct_groups = tuple(np.unique(raw_file[field]))
        n_groups = len(distinct_groups)
        set_diff_coref = list(set(np.unique(coref[field]))-
                                set(np.unique(no_coref[field])))
        # Categories in coref that are not in no_coref
        if set_diff_coref != []:
            for f in set_diff_coref:
                grouped_app = pd.DataFrame([[f,-1,0]],
                                columns=[field,'label',0])
                grouped = grouped.append(grouped_app)
        
        set_diff_no_coref = list(set(np.unique(no_coref[field]))-
                                set(np.unique(coref[field])))
        # Categories in no_coref that are not in coref
        if set_diff_no_coref != []:
            for f in set_diff_no_coref:
                grouped_app = pd.DataFrame([[f,1,0]],
                                columns=[field,'label',0])
                grouped = grouped.append(grouped_app)

        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.4
        rects1 = plt.bar(index, grouped[grouped['label']==-1][0], 
                         bar_width,
                         alpha=opacity,
                         color='b',
                         label='Non Coreference')
        rects2 = plt.bar(index + bar_width, 
                         grouped[grouped['label']==1][0], 
                         bar_width,
                         alpha=opacity,
                         color='r',
                         label='Coreference')
        plt.xlabel('Group')
        plt.ylabel('Count')
        plt.title('Count of %s by Labels' % (field))
        plt.xticks(index + bar_width, distinct_groups)
        plt.legend()

        plt.tight_layout()
        pylab.savefig('%s_dist.png' %(field), bbox_inches='tight')
        plt.close()
    else:
        no_coref = list(raw_file[raw_file['label']==-1][field])
        coref = list(raw_file[raw_file['label']==1][field])
        h = 2*iqr(raw_file[field])*(n**(-1.0/3))
        min_val = min(raw_file[field])
        max_val = max(raw_file[field])
        if h != 0:
            # Freedman-Diaconis rule
            n_bins = round((max_val-min_val)/h)
            bins = np.linspace(min_val, max_val, n_bins)
        else:
            bins = np.linspace(min_val, max_val, 5)
        plt.hist(no_coref, bins, alpha=0.5, label='Non Coreference')
        plt.hist(coref, bins, alpha=0.5, label='Coreference')
        plt.xlabel(field)
        plt.ylabel('Count')
        plt.title('Count of %s by Labels' % (field))
        plt.legend(loc='upper right')
        pylab.savefig('%s_dist.png' % (field), bbox_inches='tight')
        plt.close()