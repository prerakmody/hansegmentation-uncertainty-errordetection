# Import private libraries
import src.config as config
from src.model.trainer import Trainer,Validator

# Import public libraries
import os
import pdb
import tensorflow as tf
from pathlib import Path



if __name__ == "__main__":

    # Step1 - Define experiment name
    exp_name = 'HansegmentationUncertaintyErroDetection-OrganNet25D-CE'

    # Step 2 - Define dataloader
    data_dir = Path(config.DATA_DIR)

    resampled    = True # to homogenize all CT volumes to a fixed voxel spacing
    crop_init    = True # cropping around an organ to reduce dimensions of CT scan
    grid         = True # to perform patch-based sampling
    centred_prob = 0.3  # probability of sampling patches centred around an organ
    
    batch_size = 2
    shuffle    = 10 # buffer size to pick a random patch from
    prefetch_batch = 3 

    # Step 3 - Define model type
    model   = config.MODEL_FOCUSNET_FLIPOUT 
    mc_runs = 5
    epochs = 1500
    epochs_save = 100
    epochs_eval = 50
    epochs_viz  = 500

    # Step 4 - Build .json object for training
    params = {
        config.KEY_EXP_NAME: exp_name
        , config.KEY_RANDOM_SEED: 42
        , config.KEY_DATALOADER:{
            config.KEY_DATA_DIR: data_dir
            , config.KEY_DIR_TYPE : [config.DATALOADER_MICCAI2015_TRAIN, config.DATALOADER_MICCAI2015_TRAIN_ADD]
            , config.KEY_RESAMPLED: resampled
            , config.KEY_CROP_INIT     : crop_init
            , config.KEY_GRID          : grid
            , config.KEY_RANDOM_GRID   : True
            , config.KEY_FILTER_GRID   : False
            , config.KEY_CENTRED_PROB  : centred_prob
            , config.KEY_BATCH_SIZE    : batch_size  
            , config.KEY_SHUFFLE       : shuffle
            , config.KEY_PREFETCH_BATCH: prefetch_batch  
            , config.KEY_PARALLEL_CALLS: tf.data.AUTOTUNE  
        }
        , config.KEY_MODEL: {
            config.KEY_MODEL_NAME: model
            , config.KEY_DEEPSUP: False
            , config.KEY_KL_ALPLHA_INIT : 0.01  
            , config.KEY_KL_SCHEDULE    : config.KL_DIV_FIXED # [config.KL_DIV_FIXED, config.KL_DIV_ANNEALING]
            , config.KEY_OPTIMIZER : config.OPTIMIZER_ADAM
            , config.KEY_INIT_LR   : 0.001 
            , config.KEY_FIXED_LR  : True 
            , config.KEY_MC_RUNS   : mc_runs
            , config.KEY_EPOCHS     : epochs
            , config.KEY_EPOCHS_SAVE: epochs_save   
            , config.KEY_EPOCHS_EVAL: epochs_eval 
            , config.KEY_EPOCHS_VIZ : epochs_viz
            , config.KEY_LOAD_MODEL:{
                config.KEY_LOAD_MODEL_BOOL:False, config.KEY_LOAD_EXP_NAME: None,  config.KEY_LOAD_EPOCH:-1, config.KEY_LOAD_OPTIMIZER_LR:None
            }
            , config.KEY_MODEL_PROFILER: {
                config.KEY_PROFILE_BOOL: False
                , config.KEY_PROFILE_EPOCHS: [2,3]
                , config.KEY_PROFILE_STEPS_PER_EPOCH: 60
                , config.KEY_PROFILE_STARTING_STEP: 4
            }
            , config.KEY_MODEL_TBOARD: False
        }
        , config.KEY_METRICS : {
            config.KEY_METRIC_TBOARD: True
            # for full 3D volume
            , config.KEY_METRICS_EVAL: {'Dice': config.LOSS_DICE}
            ## for smaller grid/patch
            , config.KEY_METRICS_LOSS  : {'CE': config.LOSS_CE} # [config.LOSS_CE, config.LOSS_DICE]
            , config.KEY_LOSS_WEIGHTED : {'CE': True}
            , config.KEY_LOSS_MASK     : {'CE': True}
            , config.KEY_LOSS_COMBO    : {'CE': 1.0}
            , config.KEY_LOSS_EPOCH    : {'CE': 1.0}
            , config.KEY_LOSS_RATE     : {'CE': 1}
        }
        , config.KEY_OTHERS: {
            'epochs_timer': 20
            , 'epochs_memory':5
        }
    }

    # Call the trainer
    trainer = Trainer(params)
    trainer.train()

    # To evaluate on MICCAI2015
    params = {
        'exp_name': exp_name
        , 'pid'           : os.getpid()
        , 'dataloader': {
            'data_dir'      : data_dir
            , 'resampled'     : resampled
            , 'grid'          : grid
            , 'crop_init'     : crop_init
            , 'batch_size'    : batch_size
            , 'prefetch_batch': 1
            , 'dir_type' : [config.DATALOADER_MICCAI2015_TEST] # [config.DATALOADER_MICCAI2015_TESTONSITE]
            , 'eval_type' : config.MODE_TEST
        }
        , 'model': {
            'name': model
            , 'load_epoch'    : 1000
            , 'MC_RUNS'       : 30
            , 'training_bool' : True # [True=dropout-at-test-time, False=no-dropout-at-test-time]
        }
        , 'save': True
    }