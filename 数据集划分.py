import os
import shutil

def split_by_timeframes(frame_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    按时间片段划分视频帧为训练集、验证集和测试集
    :param frame_dir: 存储所有帧的目录
    :param output_dir: 输出目录
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "训练集、验证集和测试集的比例之和必须等于1"
    
    frames = sorted(os.listdir(frame_dir))  # 按顺序读取帧
    total_frames = len(frames)


    # 计算各个数据集的边界
    train_cutoff = int(total_frames * train_ratio)
    val_cutoff = int(total_frames * (train_ratio + val_ratio))

    # 分配时间片段
    splits = {
        "train": (0, train_cutoff),
        "val": (train_cutoff, val_cutoff),
        "test": (val_cutoff, total_frames)
    }

    # 创建输出目录并复制帧
    for split_name, (start_segment, end_segment) in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for frame_idx in range(start_segment, end_segment):
            if frame_idx < end_segment:  # 确保索引不越界
                src_path = os.path.join(frame_dir, frames[frame_idx])
                dst_path = os.path.join(split_dir, frames[frame_idx])
                shutil.copy(src_path, dst_path)

    print("时间片段划分完成！")


# 使用示例
split_by_timeframes("G:\\智能实验室项目\\dataset\\zxf", "G:\\智能实验室项目\\dataset\\output_dataset_zxf")