import torch
from torch.autograd import Function
from copy import deepcopy
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            output = self.model(input.unsqueeze(1).float()).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
    
    def dpenalty(self, model: nn.Module):
        dloss = 0
        for n, p in model.named_parameters():
            _dloss = self._precision_matrices[n] * 2 * (p - self._means[n])
            dloss += _dloss.sum()
        return dloss        

class incremental_loss(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    
    def forward(ctx, Output, Target, importance, model, old_tasks):

        InD_loss_value = F.cross_entropy(Output, Target)
        EWC_loss_value = EWC(model, old_tasks).penalty(model)
        #ctx.save_for_backward(InD_input, InD_label_onehot, OOD_input, all_class_onehot, beta, OOD_ind, InD_ind, C)
        
        loss_value = InD_loss_value + importance * EWC_loss_value
        
        return loss_value

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        #InD_input, InD_label_onehot, OOD_input, all_class_onehot, beta, OOD_ind, InD_ind, C, min_idx_OOD, min_idx_all = ctx.saved_tensors
        InD_input, InD_label_onehot, OOD_input, all_class_onehot, beta, OOD_ind, InD_ind, C= ctx.saved_tensors
        
        InD_batch_size = InD_input.shape[0]
        
        
        OOD_loss = SamplesLoss("sinkhorn", p=2, blur=1., potentials=True)
        OOD_batch_size = OOD_input.shape[0]
        OOD_f, OOD_g = OOD_loss(OOD_input[:,:,0], OOD_input, 
                                all_class_onehot[0:1].repeat(OOD_batch_size,1,1)[:,:,0],
                                all_class_onehot[0:1].repeat(OOD_batch_size,1,1))
        
        #print(OOD_ind, InD_ind)
        grad_Input = torch.zeros([InD_batch_size+OOD_batch_size, C]).to('cuda')
        
        grad_Input[OOD_ind,:] = -beta * OOD_f
        grad_Input[InD_ind,:] = -InD_label_onehot * (1. / InD_batch_size)
        
        grad_Input = EWC(model, old_tasks).dpenalty(model)
        
        return grad_Input, None, None, None, None