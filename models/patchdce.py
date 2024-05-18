from packaging import version
import torch
from torch import nn


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, weight=None):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # positive logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # weighted by semantic relation
        if weight is not None:
            l_neg_curbatch *= weight

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]# 创建只有对角线为true的对角矩阵
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        logits = (l_neg-l_pos)/self.opt.nce_T
        v = torch.logsumexp(logits, dim=1)
        loss_vec = torch.exp(v-v.detach())

        # for monitoring
        out_dummy = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        CELoss_dummy = self.cross_entropy_loss(out_dummy, torch.zeros(out_dummy.size(0), dtype=torch.long, device=feat_q.device))

        loss = loss_vec.mean()-1+CELoss_dummy.detach()

        return loss


    def forward_only_neg(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # neg logit
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = l_neg / self.opt.nce_T
        out =  torch.sigmoid(out)

        loss = self.mse_loss(out,torch.zeros(1, dtype=torch.float,
                                                        device=feat_q.device))
        return loss

    def forward_only_pos(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)


        out = l_pos

        loss = self.mse_loss(out, torch.ones(out.size(0), dtype=torch.float,
                                                        device=feat_q.device))

        return loss