import torch.nn as nn
import torch
import torch.nn.functional as F##############


class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        dim_last: int = 64,
        outputs: dict = {'bev': [0, 1]}
    ):
        super().__init__()

        dim_total = 0
        dim_max = 0

        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        dim_max = dim_max
        dim_total =  dim_total
        assert dim_max == dim_total

        self.encoder = encoder
        self.decoder = decoder
        self.outputs = outputs
        #self.proj = ProjectionHead(dim_in=self.decoder.out_channels, proj_dim=256, proj='convmlp')

        outchannels = sum(self.decoder.blocks)

        self.to_logits = nn.Sequential(
            nn.Conv2d(outchannels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, dim_max, 1))

    def forward(self, batch):
        x = self.encoder(batch)
        y = self.decoder(x)

        #########################
        y.reverse()
        output_size = y[0].size()[2:]
        y_list =[y[0]]
        for i in range(1, len(y)):
            y_list.append(nn.functional.interpolate(y[i], output_size, mode='bilinear', align_corners=True))
            
        y_out = torch.cat(y_list, 1)

        #embdding = self.proj(y[0])

        z = self.to_logits(y_out)

        z_out = {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}
        #z_out['embedding'] = embdding

        return z_out#{k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp'):
        super(ProjectionHead, self).__init__()


        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)
