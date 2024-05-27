import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List
from cross_view_transformer.model import resnet_d as Resnet_Deep
import torchgeometry as tgm
from torchgeometry.core.conversions import deg2rad
import numpy as np
import torchvision.transforms as transforms


ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        decoder_blocks: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))

        # bev coordinates
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')                      # 3 (h w)
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer('grid', grid, persistent=False)                    # 3 h w
        self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))    # d h w

    def get_prior(self):
        return self.learned_features


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        _, _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)
        dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z


class CrossViewAttention(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
        self,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """
        b, n, _, _, _ = feature.shape

        pixel = self.image_plane                                                # b n 3 h w
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]                                                     # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
        cam = I_inv @ pixel_flat                                                # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)
        d = E_inv @ cam                                                         # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 h w
        d_embed = self.img_embed(d_flat)                                        # (b n) d h w

        img_embed = d_embed - c_embed                                           # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w

        world = bev.grid[:2]                                                    # 2 H W
        w_embed = self.bev_embed(world[None])                                   # 1 d H W
        bev_embed = w_embed - c_embed                                           # (b n) d H W
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d H W
        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)      # b n d H W

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')               # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)              # (b n) d h w
        else:
            key_flat = img_embed                                                # (b n) d h w

        val_flat = self.feature_linear(feature_flat)                            # (b n) d h w

        # Expand + refine the BEV embedding
        query = query_pos + x[:, None]                                          # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w

        return self.cross_attend(query, key, val, skip=x if self.skip else None)

def bev2cam(trans_bev, f_shape):
    b, c, h, w = trans_bev.shape
    #print(trans_bev.shape)
    device = trans_bev.device
    
    crop_bev = trans_bev[:, :, :(h//2), :]
    _, _, h_c, w_c = crop_bev.shape

    rec = []

    for i in range(55, 125, 7):
        for j in range(h//2, 0, -10):
            ang = i*torch.ones(1).to(device)
            heig = j*torch.ones(1).to(device)
            h_t1 = torch.ceil(heig * torch.sin(deg2rad(ang)))
            w_t1 = torch.ceil(heig * torch.cos(deg2rad(ang)))
            #up_left =[(h_c - h_t1) , (w_c/2 - w_t1)]
            up_left = torch.unsqueeze(torch.cat([(w_c/2 - w_t1), (h_c - h_t1)]), 0)
            #print(up_left.shape)
            
            h_t2 = torch.ceil((heig-10) * torch.sin(deg2rad(ang)))
            w_t2 = torch.ceil((heig-10) * torch.cos(deg2rad(ang)))
            #down_left =[(h_c - h_t2) , (w_c/2 - w_t2)]
            #down_left =[(w_c/2 - w_t2), (h_c - h_t2)]
            if j==10:
                down_left = torch.unsqueeze(torch.cat([(w_c/2 - w_t2), (h_c - h_t2 - 1)]), 0)
            else:
                down_left = torch.unsqueeze(torch.cat([(w_c/2 - w_t2), (h_c - h_t2)]), 0)
            
            h_t3 = torch.ceil((heig-10) * torch.sin(deg2rad(ang+7)))
            w_t3 = torch.ceil((heig-10) * torch.cos(deg2rad(ang+7)))
            #down_right =[(h_c - h_t3) , (w_c/2 - w_t3)]
            #down_right =[(w_c/2 - w_t3), (h_c - h_t3)]
            if j==10:
               down_right = torch.unsqueeze(torch.cat([(w_c/2 - w_t3 + 1), (h_c - h_t3 - 1)]), 0)
            else:
               down_right = torch.unsqueeze(torch.cat([(w_c/2 - w_t3), (h_c - h_t3)]),0)
            
            h_t4 = torch.ceil((heig) * torch.sin(deg2rad(ang+7)))
            w_t4 = torch.ceil((heig) * torch.cos(deg2rad(ang+7)))
            #up_right =[(h_c - h_t4) , (w_c/2 - w_t4)]
            up_right = torch.unsqueeze(torch.cat([(w_c/2 - w_t4), (h_c - h_t4)]),0)
            
            #print(up_right.shape)
            # pts_t=torch.cat([up_left, up_left], 1)
            # pts_t=torch.cat([pts_t, down_right], 1)
            pts1=torch.cat((up_left, down_left, down_right, up_right), 0)
            #print(pts1)
            #pts1 = torch.from_numpy(np.float32([up_left, down_left, down_right, up_right]))
            pts1 = repeat(pts1, '... -> b ...', b=b)
            
            pts2 = torch.from_numpy(np.float32([[0,0], [0, np.ceil(220/10)], [np.ceil(480/10), np.ceil(220/10)], [np.ceil(480/10), 0]]))
            #print(pts2)
            pts2 = repeat(pts2, '... -> b ...', b=b).to(device)
            
            
            M2 = tgm.get_perspective_transform(pts1, pts2)
            dst2 = tgm.warp_perspective(crop_bev, M2, dsize=(22, 48))
            rec.append(dst2)
    
    out = torch.zeros(b, c, 220, 480).to(device)
    #print(out.shape)
    #print(rec[0].shape)
    #print(len(rec))
    for i in range(10):
        for j in range(10):
            out[:, :, i*22 : (i+1)*22 , (j*48): (j+1)*48] = rec[j*10+i]
    
    out = transforms.Resize((f_shape[2], f_shape[3]))(out)
    return out

def bev2cam_rear(trans_bev, f_shape):
    b, c, h, w = trans_bev.shape
    device = trans_bev.device

    crop_bev = trans_bev[:, :, :(h//2), :]
    _, _, h_c, w_c = crop_bev.shape

    rec = []

    for i in range(35, 145, 11):
        for j in range(h//2, 0, -10):
            ang = i*torch.ones(1).to(device)
            heig = j*torch.ones(1).to(device)
            h_t1 = torch.ceil(heig * torch.sin(deg2rad(ang)))
            w_t1 = torch.ceil(heig * torch.cos(deg2rad(ang)))
            #up_left =[(h_c - h_t1) , (w_c/2 - w_t1)]
            up_left = torch.unsqueeze(torch.cat([(w_c/2 - w_t1), (h_c - h_t1)]), 0)
            #print(up_left.shape)
            
            h_t2 = torch.ceil((heig-10) * torch.sin(deg2rad(ang)))
            w_t2 = torch.ceil((heig-10) * torch.cos(deg2rad(ang)))
            #down_left =[(h_c - h_t2) , (w_c/2 - w_t2)]
            #down_left =[(w_c/2 - w_t2), (h_c - h_t2)]
            if j==10:
                down_left = torch.unsqueeze(torch.cat([(w_c/2 - w_t2), (h_c - h_t2 - 1)]), 0)
            else:
                down_left = torch.unsqueeze(torch.cat([(w_c/2 - w_t2), (h_c - h_t2)]), 0)
            
            h_t3 = torch.ceil((heig-10) * torch.sin(deg2rad(ang+11)))
            w_t3 = torch.ceil((heig-10) * torch.cos(deg2rad(ang+11)))
            #down_right =[(h_c - h_t3) , (w_c/2 - w_t3)]
            #down_right =[(w_c/2 - w_t3), (h_c - h_t3)]
            if j==10:
               down_right = torch.unsqueeze(torch.cat([(w_c/2 - w_t3 + 1), (h_c - h_t3 - 1)]), 0)
            else:
               down_right = torch.unsqueeze(torch.cat([(w_c/2 - w_t3), (h_c - h_t3)]),0)
            
            h_t4 = torch.ceil((heig) * torch.sin(deg2rad(ang+11)))
            w_t4 = torch.ceil((heig) * torch.cos(deg2rad(ang+11)))
            #up_right =[(h_c - h_t4) , (w_c/2 - w_t4)]
            up_right = torch.unsqueeze(torch.cat([(w_c/2 - w_t4), (h_c - h_t4)]),0)
            
            #print(up_right.shape)
            # pts_t=torch.cat([up_left, up_left], 1)
            # pts_t=torch.cat([pts_t, down_right], 1)
            pts1=torch.cat((up_left, down_left, down_right, up_right), 0)
            #print(pts1)
            #pts1 = torch.from_numpy(np.float32([up_left, down_left, down_right, up_right]))
            pts1 = repeat(pts1, '... -> b ...', b=b)
            
            pts2 = torch.from_numpy(np.float32([[0,0], [0, np.ceil(220/10)], [np.ceil(480/10), np.ceil(220/10)], [np.ceil(480/10), 0]]))
            #print(pts2)
            pts2 = repeat(pts2, '... -> b ...', b=b).to(device)
            
            
            M2 = tgm.get_perspective_transform(pts1, pts2)
            dst2 = tgm.warp_perspective(crop_bev, M2, dsize=(22, 48))
            rec.append(dst2)
    
    out = torch.zeros(b, c, 220, 480).to(device)
    for i in range(10):
        for j in range(10):
            out[:, :, i*22 : (i+1)*22 , (j*48): (j+1)*48] = rec[j*10+i]
    
    out = transforms.Resize((f_shape[2], f_shape[3]))(out)
    return out


def inver_transfrom(bev_feature, f_shape):
    b, c, h, w = bev_feature.shape
    device = bev_feature.device
    
    center = torch.FloatTensor([h/2-1,w/2-1]).to(device)#########1*2
    center = repeat(center, '... -> b ...', b=b)
    
    angle =  -55.*torch.ones(b).to(device)
    scale = torch.ones(b).to(device)
    M1 = tgm.get_rotation_matrix2d(center,angle,scale)
    dst1 = tgm.warp_affine(bev_feature, M1, (h,w), padding_mode='reflection')
    dst2_0 = bev2cam(dst1, f_shape)
    
    angle =  torch.zeros(b).to(device)
    scale = torch.ones(b).to(device)
    M1 = tgm.get_rotation_matrix2d(center,angle,scale)
    dst1 = tgm.warp_affine(bev_feature, M1, (h,w), padding_mode='reflection')
    dst2_1 = bev2cam(dst1, f_shape)
    
    angle =  55.*torch.ones(b).to(device)
    scale = torch.ones(b).to(device)
    M1 = tgm.get_rotation_matrix2d(center,angle,scale)
    dst1 = tgm.warp_affine(bev_feature, M1, (h,w), padding_mode='reflection')
    dst2_2 = bev2cam(dst1, f_shape)
    
    angle =  -110.*torch.ones(b).to(device)
    scale = torch.ones(b).to(device)
    M1 = tgm.get_rotation_matrix2d(center,angle,scale)
    dst1 = tgm.warp_affine(bev_feature, M1, (h,w), padding_mode='reflection')
    dst2_3 = bev2cam(dst1, f_shape)
    
    
    angle =  180.*torch.ones(b).to(device)
    scale = torch.ones(b).to(device)
    M1 = tgm.get_rotation_matrix2d(center,angle,scale)
    dst1 = tgm.warp_affine(bev_feature, M1, (h,w), padding_mode='reflection')
    dst2_4 = bev2cam_rear(dst1, f_shape)
    
    
    
    angle =  110.*torch.ones(b).to(device)
    scale = torch.ones(b).to(device)
    M1 = tgm.get_rotation_matrix2d(center,angle,scale)
    dst1 = tgm.warp_affine(bev_feature, M1, (h,w), padding_mode='reflection')
    dst2_5 = bev2cam(dst1, f_shape)
    
    dst = torch.zeros(b, 6, c, f_shape[2], f_shape[3]).to(device)
    #for j in range(b):
    dst[:, 0, :, :, :]= dst2_0
    dst[:, 1, :, :, :]= dst2_1
    dst[:, 2, :, :, :]= dst2_2
    dst[:, 3, :, :, :]= dst2_3
    dst[:, 4, :, :, :]= dst2_4
    dst[:, 5, :, :, :]= dst2_5
    
    
    
    
    return dst
    
class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, dim = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))
        
    def forward(self, x):
        x_out = self.conv(x)
        
        return x_out
    
    

class Encoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            middle: List[int] = [2, 2, 2],
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        #self.backbone = backbone
        
        resnet = Resnet_Deep.resnet(pretrained=True)
        #resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3 = resnet.layer1, resnet.layer2, resnet.layer3
        #, self.layer4 = \
        #    resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        del resnet

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        #assert len(self.backbone.output_shapes) == len(middle)
        self.output_shapes = [[1, 64, 56, 120], [1, 128, 28, 60], [1, 256, 14, 30]]

        cross_views = list()
        layers = list()

        #for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
        for feat_shape, num_layers in zip(self.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        #self.in_tran = inver_transfrom()
        self.upconv1 = upconv(128, 64, 128)
        self.upconv2 = upconv(128, 128, 128)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)            # b n c h w
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4
        
        
        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W
        
        feature1 = self.layer1(self.layer0(self.norm(image)))
        feature1_bev = rearrange(feature1, '(b n) ... -> b n ...', b=b, n=n)
        x1_bev = self.cross_views[0](x, self.bev_embedding, feature1_bev, I_inv, E_inv)
        x2 = inver_transfrom(self.upconv1(x1_bev), self.output_shapes[0])
        
        x1_bev_r = self.layers[0](x1_bev)
        
        
        x2=x2.flatten(0, 1)
        feature2 = self.layer2(x2)
        feature2_bev = rearrange(feature2, '(b n) ... -> b n ...', b=b, n=n)
        x2_bev = self.cross_views[1](x1_bev_r, self.bev_embedding, feature2_bev, I_inv, E_inv)
        x3 = inver_transfrom(self.upconv2(x2_bev), self.output_shapes[1])
        
        x2_bev_r = self.layers[1](x2_bev)
        
        
        x3 = x3.flatten(0, 1)
        feature3 = self.layer3(x3)
        feature3_bev = rearrange(feature3, '(b n) ... -> b n ...', b=b, n=n)
        x3_bev = self.cross_views[2](x2_bev_r, self.bev_embedding, feature3_bev, I_inv, E_inv)
        
        x3_bev_r = self.layers[2](x3_bev)
        #x4 = inver_transfrom(x3_bev_r)
        
        # feature4 = self.layer4(x4)
        # feature4_bev = rearrange(feature4, '(b n) ... -> b n ...', b=b, n=n)
        # x4_bev = self.cross_views[3](x3_bev_r, self.bev_embedding, feature4_bev, I_inv, E_inv)
        # x4_bev_r = self.layers[3](x4_bev)
        

        # features = [self.down(y) for y in self.backbone(self.norm(image))]
        # for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
        #     feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

        #     x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv)
        #     x = layer(x)

        return x3_bev_r
