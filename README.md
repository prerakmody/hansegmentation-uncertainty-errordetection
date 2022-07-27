# Bayesian Uncertainty for Quality Assessment of Deep Learning Contours
This repository contains Tensorflow2.9 code for the paper(s)
 - Improving Error Detection in Deep Learning based Radiotherapy Autocontouring using Bayesian Uncertainty, MICCAI-UNSURE Workshop 2022

# Abstract
Bayesian Neural Nets (BNN) are increasingly used for robust organ auto-contouring. Uncertainty heatmaps extracted from BNNs have been shown to correspond to inaccurate regions. To help speed up the mandatory quality assessment (QA) of contours in radiotherapy, these heatmaps could be used as stimuli to direct visual attention of clinicians to potential inaccuracies. In practice, this is non-trivial to achieve since many accurate regions also exhibit uncertainty. To influence the output uncertainty of a BNN, we propose a modified accuracyversus-uncertainty (AvU) metric as an additional objective during model training that penalizes both accurate regions exhibiting uncertainty as well as inaccurate regions exhibiting certainty. For evaluation, we use an uncertainty-ROC curve that can help differentiate between Bayesian models by comparing the probability of uncertainty in inaccurate versus accurate regions. We train and evaluate a FlipOut BNN model on the MICCAI2015 Head and Neck Segmentation challenge dataset and on the DeepMind-TCIA dataset, and observed an increase in the AUC of uncertainty-ROC curves by 5.6% and 5.9%, respectively, when using the AvU objective. The AvU objective primarily reduced false positives regions (uncertain and accurate), drawing less visual attention to these regions, thereby potentially improving the speed of error detection.

# Model Workflow
The image below shows the the steps involved in a forward pass. 
![Model Forward Pass](/src/assets/paper_overview-v3.png)

# Results
The image below shows some results upon using the AvU (Flipout-A) and p(u|a) (FlipOut-AP) loss
![Model Forward Pass](/src/assets/MICCAI22-Flips-Ent_norm-thresh05-v3.png)

## Installation
1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/)
2. Install [git](https://git-scm.com/downloads)
3. Open a terminal and follow the commands
    - Clone this repository
        - `git clone git@github.com:prerakmody/hansegmentation-uncertainty-errordetection.git`
    - Create conda env
        - (Specifically For Windows): `conda init powershell` (and restart the terminal)
        - (For all plaforms)
        ```
        cd hansegmentation-uncertainty-errordetection
        conda deactivate
        conda create --name hansegmentation-uncertainty-errordetection python=3.8
        conda activate hansegmentation-uncertainty-errordetection
        conda develop .  # check for conda.pth file in $ANACONDA_HOME/envs/hansegmentation-uncertainty-errordetection/lib/python3.8/site-packages
        ```
    - Install packages
        - Tensorflow (check [here]((https://www.tensorflow.org/install/source#tested_build_configurations)) for CUDA/cuDNN requirements)
            - (stick to the exact commands) 
            - For tensorflow2.9
            ```
            conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
            pip install tensorflow==2.9
            ```
            - Check tensorflow installation
            ```
            python -c "import tensorflow as tf;print('\n\n\n====================== \n GPU Devices: ',tf.config.list_physical_devices('GPU'), '\n======================')"
            python -c "import tensorflow as tf;print('\n\n\n====================== \n', tf.reduce_sum(tf.random.normal([1000, 1000])), '\n======================' )"
            ```
                - [unix] upon running either of the above commands, you will see tensorflow searching for library files like libcudart.so, libcublas.so, libcublasLt.so, libcufft.so, libcurand.so, libcusolver.so, libcusparse.so, libcudnn.so in the location `$ANACONDA_HOME/envs/hansegmentation-uncertainty-errordetection/lib/`
                - [windows] upon running either of the above commands, you will see tensorflow searching for library files like cudart64_110.dll ... and so on in the location `$ANACONDA_HOME\envs\hansegmentation-uncertainty-errordetection\Library\bin`

            - Other tensorflow pacakges
            ```
            pip install tensorflow-probability==0.17.0 tensorflow-addons==0.17.1
            ```
        - Other packages
            ```
            pip install scipy seaborn tqdm psutil humanize pynrrd pydicom SimpleITK scikit-image itk-elastix scikit-learn
            pip install psutil humanize pynvml nvitop
            ```

# Notes
 - Download the data from the releases section on this repo and copy it in to the `_data/` directory
 - All the `src/{}.py` files are the backend code to train/validate/analyze the models.
 - Running any of the `demo/` folder files will train and validate either:
    - OrganNet2.5 Model (with flipout layers) + CE Loss
    - OrganNet2.5 Model (with flipout layers) + CE + AvU Loss
    - OrganNet2.5 Model (with flipout layers) + CE + AvU + p(u|a) Loss