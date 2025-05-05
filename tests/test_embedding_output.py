#!/usr/bin/env python3
# tests/test_embedding_output.py

"""
测试脚本：验证说话人嵌入提取后的维度是否正确
此脚本将从 data/processed/embeddings/ 目录加载指定的 .npy 文件（默认为 example.npy），
并打印数组的维度信息。期望的维度为 (256,)，以验证嵌入提取的正确性。
"""

import os
import sys
import numpy as np


def main():
    # 定义默认的说话人嵌入文件路径
    # 从脚本所在位置回退到项目根目录
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    default_file = "dingzhen_8.npy"
    embeddings_dir = os.path.join(base_dir, "data", "processed", "embedding")
    file_path = os.path.join(embeddings_dir, default_file)

    # 输出提示信息
    print(f"加载说话人嵌入文件: {file_path}")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print("错误：指定的嵌入文件不存在，请检查路径是否正确。")
        sys.exit(1)

    try:
        # 使用 NumPy 加载 .npy 文件
        embedding = np.load(file_path)
    except Exception as e:
        print("错误：无法加载嵌入文件。", e)
        sys.exit(1)

    # 获取并打印嵌入的维度信息
    dimension = embedding.shape
    print(f"嵌入数组的维度为: {dimension}")

    # 验证维度是否符合预期 (256,)
    expected_dim = (256,)
    if dimension == expected_dim:
        print("测试通过：嵌入维度正确 (256,)。")
        sys.exit(0)
    else:
        print(f"测试失败：嵌入维度不正确，期望 (256,) 但得到 {dimension}。")
        sys.exit(1)


if __name__ == "__main__":
    main()
