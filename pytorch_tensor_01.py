import torch

if __name__ == '__main__':
    t1=torch.tensor(7.)
    print(t1.dtype)
    t2=torch.tensor([1,2,3.])
    print(t2.shape)
    t3=torch.tensor([[1,2,3],
                    [4,5,6]])
    print(t3.shape)
    t4=torch.tensor([[[1,2,3],
                      [4,5,6]],

                     [[1,4,7],
                     [2,8,7]]])
    print(t4.shape)

    x=torch.tensor(4.)
    w=torch.tensor(5.,requires_grad=True)
    b=torch.tensor(1.,requires_grad=True)
    y=w*x+b
    print(y)
    y.backward()
    print('dy/dw=',w.grad)
    print('dy/db=',b.grad)
    print('dy/dx=',x.grad)

