"""
File: dla-sfda/configuration_test.py
Description: Configuration of your System
"""

import socket
import subprocess
import platform
import sys
import detectron2
import torch
import os


print(f"hostname: {socket.gethostname()}")
print(f"ip_address: {socket.gethostbyname(socket.gethostname())}")

info = {
    'System': platform.system(),
    'Node': platform.node(),
    'Release': platform.release(),
    'Version': platform.version(),
    'Machine': platform.machine(),
    'Processor': platform.processor()
}
print(info)

print(f"\ndetectron2: {detectron2.__version__}")

print("\nPytorch")
print(f"\ttorch: {torch.__version__}")
print(f"\ttorch.cuda: {torch.version.cuda}")
print(f"\ttorch.backends.cudnn: {torch.backends.cudnn.version()}")
if torch.cuda.is_available():
    print("\tCUDA is available")
    print("\tCUDA Device: ", torch.cuda.get_device_name(0))
else:
    print("\tCUDA not available")

print("\nPython Path:")
for path in sys.path:
    print(path)

print("\n\nEnv Variables:")
for key, value in os.environ.items():
    print(f'\t{key}: {value}')

print("\nDirectories:")
print(os.getcwd())
print(os.listdir("/"))

print(subprocess.call(["nvidia-smi"]))