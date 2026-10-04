import platform
import sys

print("=== COLAB GPU CHECK ===")
print("Python:", sys.version)
print("Platform:", platform.platform())

try:
    import torch
    import transformers
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)

    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:", torch.cuda.get_device_name(i))

        device = torch.device("cuda:0")

        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        y = torch.tensor([4.0, 5.0, 6.0], device=device)
        z = x + y

        print("Tensor device:", z.device)
        print("Tensor result:", z.tolist())
        print("GPU TEST: PASS")
    else:
        print("GPU TEST: FAIL - CUDA is not available")

except ImportError:
    print("PyTorch is not installed")
    print("GPU TEST: FAIL")
