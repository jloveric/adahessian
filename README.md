## Adahessian version using autograd.grad instead of backward
Modification to original code to use autograd.grad instead of backward (use of backward with create_graph has a memory leak).
This repo is taylored to investigation of adahessian with high-order-layers-torch.

## The original code this was based off of 

https://github.com/amirgholami/adahessian.git

## Tests
Fitting a curve with a 60th order polynomial.

Adam
```
python examples/function_example.py optimizer=adam epochs=20 n=60 optimizer.lr=1e-1
```
Adadhessian
```
python examples/function_example.py optimizer=adahessian epochs=20 n=60 optimizer.lr=1.0
```
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
