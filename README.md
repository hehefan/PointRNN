# [PointRNN: Point Recurrent Neural Network for Moving Point Cloud Processing](https://arxiv.org/pdf/1910.08287.pdf)
## PointRNN
![](https://github.com/hehefan/PointRNN/blob/master/imgs/pointrnn-arch.png)

![](https://github.com/hehefan/PointRNN/blob/master/imgs/pointrnn.png)
### Structure
![](https://github.com/hehefan/PointRNN/blob/master/imgs/units.png)
### PointGRU 
![](https://github.com/hehefan/PointRNN/blob/master/imgs/pointgru.png)
### PointLSTM
![](https://github.com/hehefan/PointRNN/blob/master/imgs/pointlstm.png)

## Moving Point Cloud Prediction 
![](https://github.com/hehefan/PointRNN/blob/master/imgs/prediction.png)

### Datasets
We provide the test sets for evaluating moving point cloud prediction:
1. [Moving MNIST Point Cloud (1 digit)](https://drive.google.com/open?id=17RpNwMLDcR5fLr0DJkRxmC5WgFn3RwK_) &emsp; 2. [Moving MNIST Point Cloud (2 digits)](https://drive.google.com/open?id=11EkVsE5fmgU5D5GsOATQ6XN17gmn7IvF) &emsp; 3. [Argoverse](https://drive.google.com/open?id=1uDsNN856IjOAOz2swfVpAjmSaEshWzHk) &emsp; 4. [nuScenes](https://drive.google.com/open?id=1nncU3D_nAjgLi_a_kxMtIaebB5WVQTDD)
### License
The code is released under MIT License.
### Citation
If you find our work useful in your research, please consider citing:
```
@article{fan19pointrnn,
  author    = {Hehe Fan and Yi Yang},
  title     = {PointRNN: Point Recurrent Neural Network for Moving Point Cloud Processing},
  journal   = {arXiv},
  volume    = {1910.08287},
  year      = {2019}
}
```
### Related Repos
1. PointRNN PyTorch implementation: https://github.com/hehefan/PointRNN-PyTorch
2. PointNet++ TensorFlow implementation: https://github.com/charlesq34/pointnet2
### Visualization
| Dataset      |    1 MNIST     |   2 MNIST  | [Argoverse](https://argoverse.org) | [Argoverse](https://argoverse.org) | [nuScenes](https://nuscenes.org)    | [nuScenes](https://nuscenes.org) |
|--------------|----------------|---------------|---------------|---------------|-------------|-------------|
| Input        | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/mmnist-1-ctx.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/mmnist-2-ctx.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/argo-bird-ctx.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/argo-worm-ctx.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/nu-bird-ctx.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/nu-worm-ctx.gif" width="100" height="100"> | 
| Ground truth | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/mmnist-1-gth.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/mmnist-2-gth.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/argo-bird-gth.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/argo-worm-gth.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/nu-bird-gth.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/nu-worm-gth.gif" width="100" height="100"> | 
| PointRNN     | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/mmnist-1-pointrnn.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/mmnist-2-pointrnn.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/argo-bird-pointrnn.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/argo-worm-pointrnn.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/nu-bird-pointrnn.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/nu-worm-pointrnn.gif" width="100" height="100"> | 
| PointGRU     | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/mmnist-1-pointgru.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/mmnist-2-pointgru.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/argo-bird-pointgru.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/argo-worm-pointgru.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/nu-bird-pointgru.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/nu-worm-pointgru.gif" width="100" height="100"> |
| PointLSTM    | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/mmnist-1-pointlstm.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/mmnist-2-pointlstm.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/argo-bird-pointlstm.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/argo-worm-pointlstm.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/nu-bird-pointlstm.gif" width="100" height="100"> | <img src="https://github.com/hehefan/PointRNN/blob/master/imgs/nu-worm-pointlstm.gif" width="100" height="100"> | 
