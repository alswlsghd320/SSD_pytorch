from numpy import sqrt
import torch
from itertools import product as product

#TODO : Convert fixed value to cfg value
class DefaultBox():
    def __init__(self, cfg):
        self.img_size = 300 #cfg['img_size']
        self.feature_maps = [38, 19, 10, 5, 3, 1] #cfg['feature_maps']
        self.ar_steps = [4, 6, 6, 6, 4, 4] #cfg['ar_steps']
        self.aspect_ratios = [1, 2, 0.5, 3, 1/3] #cfg['aspect_ratios']
        self.m = len(self.feature_maps)
        self.sk_min = 0.2 #cfg['sk_min']
        self.sk_max = 0.9 #cfg['sk_max']

        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
        if self.sk_min <= 0 or self.sk_max <= self.sk_min:
            raise ValueError('sk_min, sk_max must be grater than 0')
        # from k=1 to k=m+1=7
        self.sk = [self.compute_sk(i) for i in range(1, self.m + 2)]

    def forward(self):
        offset = []
        for k, f_k in enumerate(self.feature_maps):
            for i, j in product(range(f_k), repeat=2):
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                cnt = 0
                for ar in self.aspect_ratios:
                    cnt += 1
                    if cnt < self.ar_steps[k]:
                        if ar == 1:
                            w = h = sqrt(self.sk[k] * self.sk[k+1])
                            offset.append([cx, cy, w, h])
                        w = self.sk[k] * sqrt(ar)
                        h = self.sk[k] / sqrt(ar)
                        offset.append([cx, cy, w, h])
        return torch.Tensor(offset)

    def compute_sk(self, k):
        if k == 1:
            return self.sk_min
        elif k == self.m:
            return self.sk_max
        else:
            return round(self.sk_min + (self.sk_max - self.sk_min) / (self.m - 1) * (k - 1), 2)