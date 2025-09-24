import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QLabel, QVBoxLayout, QWidget
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# 确保能正确导入Build_Entry模块
try:
    from Build_Entry import PointMLPBackend
except ImportError as e:
    print("错误：找不到Build_Entry.py文件！请确保该文件在项目目录中")
    sys.exit(1)


class PointCloudCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # 定义13种颜色（对应0-12类）
        self.colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
            '#00FFFF', '#FFA500', '#800080', '#FFC0CB', '#A52A2A',
            '#808080', '#008000', '#000080'
        ]

    def plot_cloud(self, points, labels):
        """绘制带颜色的3D点云"""
        self.ax.clear()

        # 为每个标签选择颜色
        color_list = [self.colors[int(label)] for label in labels]

        # 绘制散点图
        self.ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=color_list, marker='o', s=5, alpha=0.8
        )

        # 设置初始视角
        self.ax.view_init(elev=30, azim=45)
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.backend = None  # 模型后端
        self.initUI()


    def initUI(self):
        self.setWindowTitle("点云语义分割工具")
        self.setGeometry(300, 300, 1000, 800)

        # 创建可视化画布
        self.canvas = PointCloudCanvas(self)

        # 创建控件
        self.btn_load = QPushButton("选择点云文件(.txt)", self)
        self.btn_load.clicked.connect(self.load_and_process_file)
        self.status_label = QLabel("就绪", self)

        # 布局管理
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)  # 画布在上方
        layout.addWidget(self.btn_load)  # 按钮在下方
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 初始化模型
        self.init_model()

    def init_model(self):
        """初始化语义分割模型"""
        try:
            # 使用绝对路径更可靠
            import os
            model_path = os.path.abspath("E:/PythonProgram/LCHNet/semantic_seg/pointnet2_sem_seg/checkpoints/best_model.pth")

            # 显示加载进度
            self.status_label.setText("正在加载模型...")
            QApplication.processEvents()  # 强制刷新界面

            self.backend = PointMLPBackend(
                model_path=model_path,
                num_classes=13,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.status_label.setText(f"模型加载成功（使用设备：{self.backend.device}）")
        except Exception as e:
            self.status_label.setText(f"模型加载失败：{str(e)}")
            raise  # 重新抛出异常以便调试

    '''输入的点云文件含有4096行，每行含有9个列 前三列为点的XYZ坐标，中间三列为点的法向量，后三列为点的归一化xyz坐标'''
    def load_and_process_file(self):
        """文件选择对话框"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择点云文件", "", "Text Files (*.txt)"
        )
        if file_path:
            self.process_file(file_path)
    '''输入读取的点云文件并处理'''
    def process_file(self, file_path):
        """处理点云文件"""
        try:
            # 读取原始数据
            self.status_label.setText("正在读取文件...")
            QApplication.processEvents()

            with open(file_path, 'r') as f:
                lines = [line.strip().split() for line in f.readlines()]
                data = [list(map(float, line)) for line in lines]

            # 转换为numpy数组
            points = np.array(data)
            if points.shape != (4096, 9):
                raise ValueError("文件格式错误：应为4096行×9列数据")

            # 模型预测
            self.status_label.setText("正在进行语义分割...")
            QApplication.processEvents()

            if self.backend is None:
                raise RuntimeError("模型未正确初始化")
            '''输出的是长度为4096的数组,也就是labels[4096]'''
            labels = self.backend.predict(points)  # 预测标签

            # 保存带标签文件
            self.status_label.setText("正在保存结果...")
            QApplication.processEvents()

            new_path = file_path.replace('.txt', '_labeled.txt')
            self.save_labeled_file(new_path, data, labels)  #  保存带标签文件

            # 可视化结果
            self.status_label.setText("正在生成可视化...")
            QApplication.processEvents()

            self.canvas.plot_cloud(points[:, :3], labels)   #  可视化
            self.status_label.setText(f"处理完成！结果保存至：{new_path}")

        except Exception as e:
            self.status_label.setText(f"错误：{str(e)}")
            print(f"错误详细信息：{str(e)}")  # 输出到控制台便于调试

    def save_labeled_file(self, path, data, labels):
        """保存带标签文件（修正后的版本）"""
        with open(path, 'w') as f:
            for row, label in zip(data, labels):
                # 修正原代码中的语法错误
                line = ' '.join(map(str, row)) + f' {int(label)}\n'
                f.write(line)


if __name__ == '__main__':
    # 配置matplotlib后端
    import matplotlib

    matplotlib.use('Qt5Agg')

    # 初始化PyTorch环境
    import torch

    print(f"PyTorch版本：{torch.__version__}")
    print(f"可用设备：{'cuda' if torch.cuda.is_available() else 'cpu'}")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())