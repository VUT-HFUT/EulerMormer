import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=2, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


########################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

###########################################################################
class MGR(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(MGR, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.dwconv1 = nn.Sequential(nn.Conv2d(hidden_features, hidden_features//3, kernel_size=1, stride=1, padding=0, groups=hidden_features//3, bias=bias), nn.GELU())
        self.dwconv3 = nn.Sequential(nn.Conv2d(hidden_features, hidden_features//3, kernel_size=3, stride=1, padding=1, groups=hidden_features//3, bias=bias), nn.GELU())
        self.dwconv5 = nn.Sequential(nn.Conv2d(hidden_features, hidden_features//3, kernel_size=5, stride=1, padding=2, groups=hidden_features//3, bias=bias), nn.GELU())
        self.act_x2 = nn.GELU()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x2 = self.act_x2(x2)
        x1_1 = self.dwconv1(x1)
        x1_3 = self.dwconv3(x1)
        x1_5 = self.dwconv5(x1)
        x1_final = torch.cat([x1_1, x1_3, x1_5], dim=1)
        x = x1_final * x2
        x = self.project_out(x)
        return x
###########################################################################
class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

##########################################################################
class DMF(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(DMF, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k,'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, C, _ = q.shape
        mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        attn_weight = (q @ k.transpose(-2, -1)) * self.temperature
        index = torch.topk(attn_weight, k=int(7), dim=-1, largest=True)[1]  # choose k
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn_weight, torch.full_like(attn_weight, float('-inf')))
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class DR_Encoder(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(DR_Encoder, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.mgr = MGR(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mgr(self.norm2(x))
        return x

class Dynamic_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Dynamic_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = DMF(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.mgr = MGR(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mgr(self.norm2(x))
        return x

class PEM(nn.Module):
    def __init__(self, dim):
        super(PEM, self).__init__()
        self.nonlinear1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = 1,padding=0, bias=False),
            nn.GELU())
        self.nonlinear2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = 1,padding=0, bias=False),
            nn.GELU())
    def forward(self, x, amp):
        mag_x = self.nonlinear2((amp - 1) * self.nonlinear1(x))
        return mag_x  

#########################################################################
class Encoder(nn.Module):
    def __init__(self,
        inp_channels, 
        dim,
        ffn_expansion_factor,
        bias = False,
        LayerNorm_type = 'WithBias'):
        super(Encoder, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.texture_encoder = nn.Sequential(*[DR_Encoder(dim=dim, num_heads =4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(2)])
        self.motion_encoder = nn.Sequential(*[DR_Encoder(dim=dim, num_heads =4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(2)])

    def forward(self, inp_img):
        x = self.patch_embed(inp_img)
        v = self.texture_encoder(x)
        m = self.motion_encoder(x) 
        return v, m # v: texture, m: shape 

class Manipulator(nn.Module):
    def __init__(self,dim,
        ffn_expansion_factor,
        bias = False,
        LayerNorm_type = 'WithBias'):
        super(Manipulator, self).__init__()
        self.dynamic_filter = nn.Sequential(*[Dynamic_TransformerBlock(dim=dim, num_heads =4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(2)])
        self.mag = PEM(dim)
        self.norm = LayerNorm(dim, LayerNorm_type)
    def forward(self, x_a, x_b, amp):
        motion = x_b - x_a
        filter_motion = self.dynamic_filter(motion)
        filter_motion = self.norm(filter_motion)
        mag_motion = self.mag(filter_motion,amp)
        return x_b + mag_motion

class Decoder(nn.Module):
    def __init__(self, 
        out_channels, 
        dim,
        ffn_expansion_factor,
        bias = False,
        LayerNorm_type = 'WithBias'):
        super(Decoder, self).__init__()
        self.refiner = nn.Sequential(*[Dynamic_TransformerBlock(dim=int(dim*2), num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(8)])
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.up = Upsample(int(dim*2))

    def forward(self, v, m): 
        recoupling = torch.cat([v,m],1) 
        refine = self.refiner(recoupling)
        refine_up = self.up(refine)
        output = self.output(refine_up) 
        return output

class MagNet(nn.Module):
    def __init__(self):
        super(MagNet, self).__init__()
        self.encoder = Encoder(inp_channels=3, dim = 48,ffn_expansion_factor = 3,bias = False,LayerNorm_type = 'WithBias')
        self.manipulator = Manipulator(dim = 48,ffn_expansion_factor = 3,bias = False,LayerNorm_type = 'WithBias')
        self.decoder = Decoder( out_channels=3, dim = 48,ffn_expansion_factor = 3,bias = False,LayerNorm_type = 'WithBias')

    def forward(self, x_a, x_b, amp, x_c, mode):
        if mode == 'train':
            v_a, m_a = self.encoder(x_a)
            v_b, m_b = self.encoder(x_b)
            v_c, m_c = self.encoder(x_c)
            m_enc = self.manipulator(m_a, m_b, amp)
            y_hat = self.decoder(v_b, m_enc)
            return y_hat, v_a, v_c, m_b, m_c
        elif mode == 'evaluate':
            v_a, m_a = self.encoder(x_a)
            v_b, m_b = self.encoder(x_b)
            motion_mag = self.manipulator(m_a, m_b ,amp)
            y_hat = self.decoder(v_b, motion_mag)
            return y_hat
