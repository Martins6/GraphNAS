import torch

def check_cuda_status():
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        current_device = torch.cuda.current_device()
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"Current CUDA Device: {current_device}")
        print(f"Total CUDA Devices: {device_count}")
        print(f"CUDA Device Name: {device_name}")
        
        memory_allocated = torch.cuda.memory_allocated(current_device)
        memory_cached = torch.cuda.memory_reserved(current_device)
        
        print(f"Allocated Memory: {memory_allocated/1024**2:.2f} MB")
        print(f"Cached Memory: {memory_cached/1024**2:.2f} MB")

if __name__ == "__main__":
    check_cuda_status()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.rand(3,3).to(device)
        print("\nTest tensor on GPU:")
        print(x)
    else:
        print("\nRunning on CPU only")
