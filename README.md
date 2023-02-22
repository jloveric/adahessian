## Adahessian version using autograd.grad instead of backward
Modification to original code to use autograd.grad instead of backward (use of backward with create_graph has a memory leak).
This repo is taylored to investigation of adahessian with high-order-layers-torch.

## The original code this was based off of 

https://github.com/amirgholami/adahessian.git


## This original paper citation is here
AdaHessian has been developed as part of the following paper. We appreciate it if you would please cite the following paper if you found the library useful for your work:

```text
@article{yao2020adahessian,
  title={ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning},
  author={Yao, Zhewei and Gholami, Amir and Shen, Sheng and Keutzer, Kurt and Mahoney, Michael W},
  journal={AAAI (Accepted)},
  year={2021}
}
```
