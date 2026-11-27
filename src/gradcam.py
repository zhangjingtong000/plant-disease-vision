# -*- coding: utf-8 -*-
"""
gradcam.py — 模型可解释性：Grad-CAM 热力图、病斑定位、遮挡实验

三件事：
1. Grad-CAM：找出模型关注区域
2. 白底屏蔽 + 病斑定位：把热力图转成"病斑框 + 面积占比"
3. 遮挡测试：遮住病斑 vs 遮住随机区域，比较置信度跌幅
"""
import numpy as np
import torch
import torch.nn.functional as F
import cv2
class GradCAM:
    """针对 torchvision 三种骨干的 Grad-CAM 实现。"""

    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.gradients = None
        self.activations = None
        self.handles = []

        # 判定目标层
        if model_name.startswith("resnet"):
            target = model.layer4[-1]
        elif model_name.startswith("efficientnet"):
            target = model.features[-1]
        elif model_name.startswith("mobilenet"):
            target = model.features[-1]
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        self.handles.append(target.register_forward_hook(self._fwd_hook))
        self.handles.append(target.register_full_backward_hook(self._bwd_hook))

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        """
        参数 x: [1, 3, H, W] 输入张量
        返回 (cam 归一化矩阵 HxW, 预测类别)
        """
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("未获取到梯度或激活，检查模型结构")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)      # [1,C,1,1]
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return cam, class_idx

    def close(self):
        for h in self.handles:
            h.remove()


def leaf_mask(rgb_uint8, white_thresh=228):
    """
    生成叶片区域掩码：纯白背景置 False，叶片置 True。
    输入为 HxWx3 的 uint8 RGB 图像。
    """
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    mask = (gray < white_thresh).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 只保留最大连通域（叶片主体）
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros(gray.shape, dtype=bool)
    c = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [c], -1, 255, thickness=-1)
    return clean > 0


def locate_lesion(cam, mask, thresh=0.6, min_area_ratio=0.0008):
    """
    把 CAM 热力图转成病斑区域。

    返回 dict:
        lesion_mask   : 病斑二值掩码（与原图同尺寸）
        bbox          : (x, y, w, h) 或 None
        area_ratio    : 病斑面积 / 叶片面积
    """
    h, w = cam.shape
    cam_masked = cam * mask.astype(np.float32)
    binary = ((cam_masked >= thresh) & mask).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_area = max(int(mask.sum()), 1)

    if not cnts:
        return {"lesion_mask": np.zeros((h, w), bool), "bbox": None, "area_ratio": 0.0}

    # 合并所有足够大的连通域
    keep = [c for c in cnts if cv2.contourArea(c) / leaf_area >= min_area_ratio]
    if not keep:
        return {"lesion_mask": np.zeros((h, w), bool), "bbox": None, "area_ratio": 0.0}

    lesion = np.zeros((h, w), np.uint8)
    cv2.drawContours(lesion, keep, -1, 1, thickness=-1)
    xs, ys, ws, hs = [], [], [], []
    for c in keep:
        x, y, ww, hh = cv2.boundingRect(c)
        xs.append(x); ys.append(y); ws.append(x + ww); hs.append(y + hh)
    bbox = (min(xs), min(ys), max(ws) - min(xs), max(hs) - min(ys))
    area_ratio = float(lesion.sum()) / leaf_area

    return {"lesion_mask": lesion.astype(bool), "bbox": bbox, "area_ratio": area_ratio}


def overlay_cam_on_image(rgb_uint8, cam, mask, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """把热力图叠加到原图，背景区域不叠加。"""
    cam_masked = (cam * mask.astype(np.float32) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam_masked, colormap)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = rgb_uint8.copy()
    idx = mask
    out[idx] = (out[idx] * (1 - alpha) + heat[idx] * alpha).astype(np.uint8)
    return out


def occlusion_test(model, x, cam, mask, device, ratio=0.35):
    """
    遮挡实验：分别遮挡 CAM 高响应区与同面积随机区域，返回置信度变化。

    参数 ratio: 遮挡面积占 CAM 响应的比例（默认 35%）

    返回 dict:
        base_conf            基线置信度
        occlude_lesion_conf  遮挡病斑后的置信度
        occlude_random_conf  遮挡随机区域后的置信度
    """
    model.eval()

    def conf_of(inp):
        with torch.no_grad():
            out = model(inp)
            p = torch.softmax(out, dim=1)[0]
        return float(p.max().item())

    base = conf_of(x)

    # 病斑区域：CAM 最高响应的 ratio 比例像素
    flat = cam.reshape(-1)
    k = max(int(flat.size * ratio), 1)
    thresh = np.partition(flat, -k)[-k]
    lesion_area = (cam >= thresh) & mask

    x_occ = x.clone()
    x_occ[0, :, lesion_area] = 0.0
    conf_lesion = conf_of(x_occ)

    # 随机区域：同尺寸，从叶片内「非病斑」区域随机选，保证对比公平
    rng = np.random.RandomState(0)
    healthy_mask = mask & (~lesion_area)
    ys, xs = np.where(healthy_mask)
    rand_area = np.zeros_like(lesion_area)
    n_pick = int(lesion_area.sum())
    if n_pick > 0 and len(ys) > n_pick:
        idx = rng.choice(len(ys), n_pick, replace=False)
        rand_area[ys[idx], xs[idx]] = True

    x_rand = x.clone()
    x_rand[0, :, rand_area] = 0.0
    conf_rand = conf_of(x_rand)

    return {
        "base_conf": base,
        "occlude_lesion_conf": conf_lesion,
        "occlude_random_conf": conf_rand,
        "drop_lesion": base - conf_lesion,
        "drop_random": base - conf_rand,
    }
