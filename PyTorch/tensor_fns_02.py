import torch
import numpy as np
if __name__=='__main__':
    t1=torch.full((3,2),40.)
    print(t1)
    t2=torch.sin(t1)
    print(t2)
    t3=torch.cat((t1,t2))
    print(t3)
    t4=t3.reshape(2,6)
    print(t4)
    # Numpy pytorch-tensor conversion
    s=np.array([1,2,3.])
    print(s.dtype)
    x=torch.from_numpy(s)
    print(x.dtype)
    z=x.numpy()
    print(z.dtype)