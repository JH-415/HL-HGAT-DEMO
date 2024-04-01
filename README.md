# HL-HGAT-DEMO
DEMO for OHBM
# HL-HGAT
Heterogeneous Graph Convolutional Neural Network via Hodge-Laplacian

<picture>
 <img alt="Model Architecture" src="Architecture_v2.png">
</picture>

This project introduces a novel approach to transform a traditional graph into a simplex graph, where nodes, edges, and higher-order interactions are characterized by different-dimensional simplices. We propose the Hodge-Laplacian Heterogeneous Graph Attention Network (HL-HGAT), which enables simultaneous learning of features on different-dimensional simplices.

In this HL-HGAT package, we provide the transformation of the original graph to a simplex graph. Then we provide a detailed implementation of the proposed model. HL-HGAT is built using PyG and Pytorch.

## Python environment setup with Conda
cu102 should be replaced by the specific CUDA versions!
```bash
conda create -n HLHGCNN python=3.9
conda activate HLHGCNN
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

pip install -y torch-cluster==1.6.0     -f https://pytorch-geometric.com/whl/torch-1.12.1+cu102.html
pip install torch-scatter==2.0.9     -f https://pytorch-geometric.com/whl/torch-1.12.1+cu102.html
pip install torch-sparse==0.6.15      -f https://pytorch-geometric.com/whl/torch-1.12.1+cu102.html
pip install torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu102.html

conda install -c conda-forge timm
conda install -c anaconda networkx
conda install -c conda-forge mat73
conda install -c conda-forge torchmetrics
conda clean --all
```

## file descriptions
./OHBM-DEMO:  Step-by-Step Guide for Implementing the HL-HGAT Model.


## Citation
If you find this work useful, please cite our IPMI 2023 paper:
```bash
@inproceedings{huang2023heterogeneous,
  title={Heterogeneous Graph Convolutional Neural Network via Hodge-Laplacian for Brain Functional Data},
  author={Huang, Jinghan and Chung, Moo K and Qiu, Anqi},
  booktitle={International Conference on Information Processing in Medical Imaging},
  pages={278--290},
  year={2023},
  organization={Springer}
}
```
If you are using MSI and SAP modules, please cite our new submission:
```
@article{huang2024advancing,
  title={Advancing Graph Neural Networks with HL-HGAT: A Hodge-Laplacian and Attention Mechanism Approach for Heterogeneous Graph-Structured Data},
  author={Huang, Jinghan and Chen, Qiufeng and Bian, Yijun and Zhu, Pengli and Chen, Nanguang and Chung, Moo K and Qiu, Anqi},
  journal={arXiv preprint arXiv:2403.06687},
  year={2024}
}
```





