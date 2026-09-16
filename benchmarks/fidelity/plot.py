#!/usr/bin/env python3
"""Plot frozen measured baselines, optionally adding a community result."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path);p.add_argument('--out',type=Path,default=Path('comparison'));a=p.parse_args()
    baseline=json.loads((ROOT/'data/baselines.json').read_text())
    rows=[{'label':name,**value} for name,value in baseline['models'].items()]
    if a.result:
        result=json.loads(a.result.read_text())
        if result['status']!='PASS' or result['panel_sha256']!=baseline['panel_sha256']:raise ValueError('Only complete results from the frozen panel can be plotted')
        rows.append(result)
    colors=['#5754d6','#df7b32','#09878a','#c43e70']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc'})
    fig,axes=plt.subplots(2,1,figsize=(13,10),sharex=True)
    fig.subplots_adjust(left=.10,right=.96,top=.80,bottom=.16,hspace=.35)
    fig.text(.10,.945,'Qwen3.8 Flash: fixed-corpus fidelity',fontsize=23,weight='bold')
    fig.text(.10,.90,'Original BF16 reference | 16,384 scored positions | Same frozen engineering text',color='#596a80')
    for i,row in enumerate(rows):fig.text(.10+i*.22,.85,row['label'],color=colors[i],weight='bold')
    for ax,key,title in zip(axes,['top1_agreement_percent','mean_kl_nats'],['Strict top-1 agreement (higher is better)','Forward KL divergence, nats (lower is better)']):
        ax.set_title(title,loc='left',weight='bold');ax.grid(axis='y',color='#dfe5ee')
        for i,row in enumerate(rows):
            m=row['metrics'][key];x=row['size_bytes']/2**30;y=m['mean'];lo,hi=m['ci95']
            ax.errorbar(x,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='D' if i==3 else 'o',color=colors[i],capsize=5,markersize=9)
            text=f'{y:.2f}%' if key.startswith('top1') else f'{y:.5f}'
            ax.annotate(text,(x,y),xytext=(8,12+i*3),textcoords='offset points',color=colors[i],weight='bold',bbox={'facecolor':'#f8fafc','edgecolor':'none','pad':1})
        ax.margins(x=.22,y=.35)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes[1].set_xlabel('Complete text target, including PLE / n-gram storage (GiB)')
    fig.text(.10,.08,'Whiskers: 95% bootstrap intervals over eight repository clusters. Not task accuracy.',fontsize=10,color='#596a80')
    fig.text(.10,.055,'Full model + runtime configurations; custom kernels can affect fidelity. No MTP or sampling.',fontsize=10,color='#596a80')
    fig.text(.10,.03,'Different corpus/reference from the publisher chart: those percentages cannot be placed on this scale.',fontsize=10,color='#596a80')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg','pdf']:fig.savefig(str(a.out)+'.'+suffix,dpi=170)
    plt.close(fig)
if __name__=='__main__':main()
