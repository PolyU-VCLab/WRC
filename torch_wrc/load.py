import os
from torch.utils.cpp_extension import load

os.environ['MAX_JOBS'] = '32' 
current_dir = os.path.dirname(os.path.abspath(__file__))

sources = [
    os.path.join(current_dir, 'weighted_reverse_convolution.cpp'),
    os.path.join(current_dir, 'sfold_upsample.cu'),
    os.path.join(current_dir, 'splits_mean.cu'),
    os.path.join(current_dir, 'p2o.cu')
]

for f in sources:
    assert os.path.exists(f), f"File {f} does not exist."

print("Compiling C++ extension, please wait...")
wrc = load(
    name='wrc', 
    sources=sources,
    extra_cflags=['-O3'],          # C++ compiler flags
    extra_cuda_cflags=['-O3', '--use_fast_math'], # CUDA compiler flags
    verbose=True,                  # Print verbose build logs for debugging
)
print("Compilation finished.")

