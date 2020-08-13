import torch

def sim(u, t):
    ut=torch.t(u).dot(t)
    un=torch.norm(u)
    tn=torch.norm(t)
    return ut/(un*tn)

temp=0.1

def loss_func(zi, zj):

    batch_size=len(zi)
    
    ## Normalize the data
    zi=torch.div(zi, torch.norm(zi, dim=1).reshape(-1,1))
    zj=torch.div(zj, torch.norm(zj, dim=1).reshape(-1,1))
    z=torch.cat([zi, zj])
    
    top_ij=torch.exp(torch.div(torch.nn.CosineSimilarity()(zi, zj),temp))
    top_ji=torch.exp(torch.div(torch.nn.CosineSimilarity()(zj, zi),temp))
    top=torch.cat([top_ij,top_ji])
    
    bottom_all=torch.exp(torch.div(torch.mm(z,torch.t(z)), temp))
    diagonal=torch.diag(torch.diagonal(bottom_all))
    bottom=torch.sum(bottom_all-diagonal,dim=1)
    
    l=-torch.log(torch.div(top, bottom))
    
    L=torch.div(torch.sum(l), 2*batch_size)
    return L
            