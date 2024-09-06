## Enhanced Visible-Infrared Person Re-Identification via Potential Representation-Based Multiple Feature Extraction Network

Official PyTorch implementation of the paper Enhanced Visible-Infrared Person Re-Identification via Potential Representation-Based Multiple Feature Extraction Network.
### 1. Prepare the datasets.
- (1) RegDB Dataset [1]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.
- (2) SYSU-MM01 Dataset [2]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).
-   run  `python pre_process_sysu.py`  to pepare the dataset, the training data will be stored in ".npy" format.

### 2. Training.
Train a model by:

```
python train.py --dataset sysu --gpu 0
```

--dataset: which dataset "sysu" or "regdb".

--gpu: which gpu to run.

You may need mannully define the data path first.

Parameters: More parameters can be found in the script.

### 3. Testing.
Test a model on SYSU-MM01 or RegDB dataset by
```
python test.py --mode all --tvsearch True --resume 'model_path' --gpu 0 --dataset sysu
```
--dataset: which dataset "sysu" or "regdb".

--mode: "all" or "indoor" all search or indoor search (only for sysu dataset).

--tvsearch: whether thermal to visible search (only for RegDB dataset).

--resume: the saved model path.

--gpu: which gpu to run.
### 4. References.
[1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

