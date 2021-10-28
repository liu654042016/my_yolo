import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import *
from models.experimental import *
from utils.autoanchor import  check_anchors
from utils.general import check_yaml, make_divisible, print_args, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import copy_attr,fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
    select_device, time_sync


try:
    import thop
except ImportError:
    thop = None
LOGGER = logging.getLogger(__name__)

class Detect(nn.Model):
    stride = None
    onnx_dynamic = False
    def __init__(self, nc = 80, anchors = () ,ch = (), inplace=True):
        super().__init__()
        self.nc = nc #number of cilass
        self.no = nc + 5 #number of outputsper anchor
        self.nl = len(anchors) #number of detection layer
        self.na = len(anchors[0]) // 2 #number of anchors
        self.grid = [torch.zeros(1)] * self.nl #init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)#shape(nl, na, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))#shape(nl, 1, na, 1, 1, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no*self.na, 1) for x in ch) #output conv
        self.inplace = inplace #use in-place ops
    def forward(self, x):
        z = [] #inference out
        for i in range(self.nl):
            x[i] = self.m[i](x[i]) #conv
            bs, _, ny, nx = x[i].shape#x(bs, 255m 20, 20) to x(bs, 3, 20, 20, 85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:#inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2]*2. - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4]*2) ** 2 * self.anchor_grid[i] #wh
                else:
                    xy = (y[..., 0:2]*2. - 0.5 + self.grid[i])*self.stride[i]
                    wh = (y[..., 2:4]*2)**2*self.anchor_grid[i].view(1, self.na, 1, 1, 2)
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class Model(nn.Module):
    def __init__(self, cfg = 'yolov5s.yaml', ch = 3, nc = None, anchors = None):#model , input channels, number 0f class
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg #model dict
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg, errors='ignore') as f:
                self.yaml = yaml.safe_load(f)#model dict

        #define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)#input channels
        if nc and nc !=self.yaml['nc']:
            LOGGER.info(f"Overriding model.ymal nc={self.yaml['nc']}with nc={nc} ")
            self.yaml['nc'] = nc
        if anchors:
            LOGGER.info(f'Overriding model.ymal anchors with anchors = {anchors}')
            self.yaml['anchors'] = round(anchors)#override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch]) #model savelist
        self.names = [str(i) for i in range(self.yaml['nc'])] #defalut name
        self.inplace = self.yaml.get('inplace', True)

        #build stides , anchors
        m = self.model[-1] #Detect
        if isinstance(m, Detect):
            s = 256 #2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]) #forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases() #only run once

        #init weights biase
        initialize_weights(self)
        self.info()
        LOGGER.info('')
    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile, visualize)
    def _forward_agument(self, x):
        img_size = x.shape[-2:] #height width
        s = [1, 0.83, 0.67] #scales
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0] #forward
            #cv2.imwrite
            yi = self._descale_pred(yi, fi, xi, img_size)
            y.append(yi)
        y = self._clip_augmented(y) #clip augmented tails
        return torch.cat(y, 1), None
    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1: #if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j==-1 else y[j] for j in m.f]#from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x
    def _descale_pred(self, p, flips, scale, img_size):
        if self.inplace:
            p[..., :4] /= scale
            if flips ==2 :
                p[..., :1] = img_size[0] - p[..., 1]
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]
        else:
            x, y, wh = p[..., 0:1]/scale, p[..., 1:2]/scale, p[..., 2:4]/scale
            if flips == 2:
                y = img_size[0] - y
            elif flips ==3:
                x = img_size[1] - x
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p