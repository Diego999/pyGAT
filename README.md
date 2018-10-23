# Pytorch Graph Attention Network

This is a pytorch implementation of the Graph Attention Network (GAT)
model presented by Veličković et. al (2017, https://arxiv.org/abs/1710.10903).

The repo has been forked initially from https://github.com/tkipf/pygcn. The official repository for the GAT (Tensorflow) is available in https://github.com/PetarV-/GAT. Therefore, if you make advantage of the pyGAT model in your research, please cite the following:

```
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
  note={accepted as poster},
}
```

The branch **master** contains the implementation from the paper. The branch **similar_impl_tensorflow** the implementation from the official Tensorflow repository.

# Performances

For the branch **master**, the training of the transductive learning on Cora task on a Titan Xp takes ~0.9 sec per epoch and 10-15 minutes for the whole training (~800 epochs). The final accuracy is between 84.2 and 85.3 (obtained on 5 different runs). For the branch **similar_impl_tensorflow**, the training takes less than 1 minute and reach ~83.0.

A small note about initial sparse matrix operations of https://github.com/tkipf/pygcn: they have been removed. Therefore, the current model take ~7GB on GRAM.

# Sparse version GAT

We develop a sparse version GAT using pytorch. There are numerically instability because of softmax function. Therefore, you need to initialize carefully. To use sparse version GAT, add flag `--sparse`. The performance of sparse version is similar with tensorflow. On a Titan Xp takes 0.08~0.14 sec.

# Requirements

pyGAT relies on Python 3.5 and PyTorch 0.4.1 (due to torch.sparse_coo_tensor).

# Issues/Pull Requests/Feedbacks

Don't hesitate to contact for any feedback or create issues/pull requests.
