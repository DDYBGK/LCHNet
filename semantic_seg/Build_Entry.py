import torch
import numpy as np
from model.pointMLP import PointMLP  # 确保能导入模型定义
from pointnet2_sem_seg.pointnet2_sem_seg import get_model  # 确保能导入模型定义

class PointMLPBackend:
    def __init__(self, model_path, num_classes=13, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # 初始化模型
        self.model = get_model(num_classes=num_classes)
        #self.model = PointMLP(num_sem=num_classes)
        
        # 加载预训练权重
        state_dict = torch.load(model_path, map_location=self.device)
        if 'model' in state_dict:  # 兼容checkpoint格式
            state_dict = state_dict['model']
        
        # 处理可能的DataParallel前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
    
    def preprocess(self, points):
        """
        输入: numpy数组 [N, 9] (XYZ+其他特征)
        输出: 模型需要的张量格式 [1, 9, 4096]
        """
        # 确保点数正确
        if points.shape[0] < 4096:
            # 填充
            pad_size = 4096 - points.shape[0]
            points = np.pad(points, ((0, pad_size), (0, 0)), mode='constant')
        elif points.shape[0] > 4096:
            # 随机下采样
            idx = np.random.choice(points.shape[0], 4096, replace=False)
            points = points[idx]
        
        # 转换为PyTorch张量并添加batch维度
        points_tensor = torch.from_numpy(points.T).float().unsqueeze(0)  # [1, 9, 4096]
        return points_tensor.to(self.device)
    
    def postprocess(self, logits):
        """
        输入: 模型输出张量 [1, 4096, 13]
        输出: 预测标签 [N]
        """
        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
        pred_labels = np.argmax(probs, axis=-1)[0]  # [4096]
        return pred_labels  #数组里边的值是0-12
    
    def predict(self, points):
        """
        完整推理流程:
        输入: 原始点云 [N, 9]
        输出: 语义标签 [N] (原始点数)
        """
        with torch.no_grad():
            # 预处理
            input_tensor = self.preprocess(points)
            
            # 推理
            logits = self.model(input_tensor)  # [1, 4096, 13]
            
            # 后处理
            full_pred = self.postprocess(logits)  #full_pred是一个数组，里面有4096个预测的标签值（0-12）
            
        # 截断到原始点数
        return full_pred

# 使用示例
if __name__ == '__main__':
    # 初始化后端
    backend = PointMLPBackend(
        model_path="pointnet2_sem_seg/checkpoints/best_model.pth",
        num_classes=13,
        device='cuda'  # 优先使用GPU
    )
    
    # 模拟输入 (N,9) 这里N=5000用于测试下采样
    dummy_points = np.random.randn(5000, 9).astype(np.float32)
    
    # 推理
    pred_labels = backend.predict(dummy_points)

    print("预测结果形状:", pred_labels.shape)
    print("示例标签:", pred_labels[:10])