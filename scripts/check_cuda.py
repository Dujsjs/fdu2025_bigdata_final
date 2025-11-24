import torch
def check_cuda_pytorch():
    print("--- PyTorch CUDA 检查 ---")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")

    if cuda_available:
        try:
            num_gpus = torch.cuda.device_count()
            print(f"可用 GPU 数量: {num_gpus}")

            # 假设使用第一个 GPU (索引 0)
            gpu_name = torch.cuda.get_device_name(0)
            print(f"第一个 GPU 名称: {gpu_name}")

            # 验证 PyTorch 是否能实际与 GPU 通信
            device = torch.device("cuda:0")
            x = torch.ones(5, 5, device=device)
            print(f"成功在 {device} 上创建张量。")

            print("\n✅ PyTorch 能够正常调用 CUDA。")

        except Exception as e:
            print(f"\n❌ PyTorch 发现 CUDA，但通信失败。错误: {e}")

    else:
        print("\n❌ CUDA 不可用。请检查环境配置。")


if __name__ == '__main__':
    check_cuda_pytorch()