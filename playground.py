import pytorch_lightning as pl
import wandb, os
from glob import glob
from logic import classifier
from base_pipe import base_pipe
from model import get_model
import pdb
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def get_folds():
    df= pd.DataFrame()
    base_img_folder = '/Users/sagar/Desktop/MachineLearningEngineerTest/ML/data/images'
    df['img_list'] = glob(base_img_folder+'/*/*.TIF')
    df['cls'] = [int(x.split('/')[-2]) for x in df['img_list']]
    df['fold'] = 0
    skf= StratifiedKFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(skf.split(df,df['cls'])):
        df.loc[test_index,'fold']= i
    # df.to_csv('sample.csv',index=False)
    return df

def main():
    wnadb_run = wandb.init(
                        project='Test_1',
                        name='exp0'
                    )

    generate_csv = True
    if generate_csv:
        df = get_folds()
    fold = 0
    num_classes = 5
    model = get_model(num_classes=num_classes)
    num_workers = 1
    Classifier = classifier(
                            df=df,
                            fold=fold,
                            model=model,
                            ds = base_pipe,
                            bs = 16,
                            num_classes=num_classes,
                            wandb_run =wnadb_run,
                            num_workers = num_workers
                            )

    Trainer=pl.Trainer(
                        # devices=1,
                        # accelerator="mps",
                        max_epochs=35,
                      )
    
    Trainer.fit(Classifier)
    return 

if __name__=="__main__":
    main()
