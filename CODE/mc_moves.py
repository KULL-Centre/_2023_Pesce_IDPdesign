import pandas as pd
import numpy as np
import random

def MC_move(df,c,move):
    seq_ref = list(df[df['mc']==True].fasta)[-1][:]
    seq = list(df[df['mc']==True].fasta)[-1][:]
    if move == 'swap_full':
        position1 = random.randint(0,len(df.fasta[df.index[-1]])-1)
        position2 = random.randint(0,len(df.fasta[df.index[-1]])-1)
    elif move == 'swap_expr':
        position1 = random.randint(1,len(df.fasta[df.index[-1]])-1)
        position2 = random.randint(1,len(df.fasta[df.index[-1]])-1)
    seq[position1] = seq_ref[position2]
    seq[position2] = seq_ref[position1]
    df.loc[df.index[-1]+1] = dict(fasta=list(seq),
                                    obs=False,
                                    simulate=False,
                                    mc=False,
                                    mc_cp=c)
    return df
