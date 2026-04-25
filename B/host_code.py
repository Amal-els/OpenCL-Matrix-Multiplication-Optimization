import numpy
import random
import pyopencl as cl
from time import time


def clamp(x, min_val, max_val):
    return min(max_val, max(min_val, x))


def read_int(prompt, fallback=None):
    try:
        return int(input(prompt))
    except ValueError, TypeError:
        return fallback


# Number of matrix multiplications to perform
COUNT = 1

# Matrix dimensions
DEFAULT_DIM = 8192
M = N = K = DEFAULT_DIM

print("N:", N)

# Number of GFLOP to be performed (multiply + add = factor of 2)
gflop = COUNT * 2.0 * M * N * K / 1e9

# Dummy data: all elements in each matrix are equal
AVAL = 3.257
BVAL = 5.723
CVAL = float(K) * AVAL * BVAL

# --- Device selection ---
use_gpu = read_int("Use GPU? (yes: 1, no: 0, default: 1): ", 1) != 0
use_integrated_gpu = read_int("Use integrated_gpu? (yes: 1, no: 0, default: 1): ", 1) != 0

if not use_gpu and not use_integrated_gpu:
    print("At least one of GPU or integrated_gpu must be enabled. Defaulting to GPU only.")
    use_gpu = True

# Partition rows between GPU and integrated_gpu
if use_gpu and use_integrated_gpu:
    GPU_M = 15 * M // 16
    integrated_gpu_M = M // 16
elif use_gpu:
    print("The GPU is going to perform all the work.")
    GPU_M = M
    integrated_gpu_M = 0
else:
    print("The integrated_gpu is going to perform all the work.")
    GPU_M = 0
    integrated_gpu_M = M

# Buffer sizes
sizeB = K * N
sizeAGPU = GPU_M * K
sizeAintegrated_gpu = integrated_gpu_M * K
sizeCGPU = GPU_M * N
sizeCintegrated_gpu = integrated_gpu_M * N

# --- Kernel configuration ---
gpu_kernel_name = "./B/gpu_kernel.cl"
integrated_gpu_kernel_name = "./B/integrated_gpu_kernel.cl"

DEFAULT_TS = 16
TS = DEFAULT_TS

if use_gpu:
    print("\nGPU kernel:", gpu_kernel_name)
    TS = read_int(f"TS (4, 8, 16, 32) (default: {DEFAULT_TS}): ", DEFAULT_TS)
    if TS not in [4, 8, 16, 32]:
        print(f"Invalid tile size. Using default: {DEFAULT_TS}")
        TS = DEFAULT_TS
    print(f"Work group size: {TS} x {TS}")

DEFAULT_WIDTH = 4
DEFAULT_TSM = 128
DEFAULT_TSN = 128
DEFAULT_TSK = 32
DEFAULT_WPTM = 8
DEFAULT_WPTN = 8

WIDTH = DEFAULT_WIDTH
TSM = DEFAULT_TSM
TSN = DEFAULT_TSN
TSK = DEFAULT_TSK
WPTM = DEFAULT_WPTM
WPTN = DEFAULT_WPTN

if use_integrated_gpu:
    print("\nintegrated_gpu kernel:", integrated_gpu_kernel_name)
    WIDTH = read_int("Work per thread (1, 2, 4) (default: 4): ", DEFAULT_WIDTH)
    if WIDTH not in [1, 2, 4]:
        print(f"Invalid width. Using default: {DEFAULT_WIDTH}")
        WIDTH = DEFAULT_WIDTH
    TSM = read_int(f"TSM  (default: {DEFAULT_TSM}):  ", DEFAULT_TSM)
    TSN = read_int(f"TSN  (default: {DEFAULT_TSN}):  ", DEFAULT_TSN)
    TSK = read_int(f"TSK  (default: {DEFAULT_TSK}):  ", DEFAULT_TSK)
    WPTM = read_int(f"WPTM (default: {DEFAULT_WPTM}): ", DEFAULT_WPTM)
    WPTN = read_int(f"WPTN (default: {DEFAULT_WPTN}): ", DEFAULT_WPTN)

# --- Host buffers ---
gpu_h_B = numpy.full(sizeB, BVAL, dtype=numpy.float32)
integrated_gpu_h_B = numpy.full(sizeB, BVAL, dtype=numpy.float32)

gpu_h_A = gpu_h_C = None
integrated_gpu_h_A = integrated_gpu_h_C = None

if use_gpu:
    gpu_h_A = numpy.full(sizeAGPU, AVAL, dtype=numpy.float32)
    gpu_h_C = numpy.empty(sizeCGPU, dtype=numpy.float32)

if use_integrated_gpu:
    integrated_gpu_h_A = numpy.full(sizeAintegrated_gpu, AVAL, dtype=numpy.float32)
    integrated_gpu_h_C = numpy.empty(sizeCintegrated_gpu, dtype=numpy.float32)

# --- GPU OpenCL setup ---
gpu_context = gpu_queue = gpu_mmul = None
gpu_a = gpu_b = gpu_c = None

if gpu_h_A is not None and gpu_h_C is not None:
    print("\nCreating a GPU OpenCL context...")
    gpu_kernel_source = f"#define TS {TS}\n" + open(gpu_kernel_name).read()
    gpu_context = cl.create_some_context()
    gpu_queue = cl.CommandQueue(gpu_context)
    gpu_a = cl.Buffer(gpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=gpu_h_A)
    gpu_b = cl.Buffer(gpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=gpu_h_B)
    gpu_c = cl.Buffer(gpu_context, cl.mem_flags.WRITE_ONLY, size=gpu_h_C.nbytes)
    gpu_program = cl.Program(gpu_context, gpu_kernel_source).build()
    gpu_mmul = gpu_program.mmul
    gpu_mmul.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])

# --- integrated_gpu OpenCL setup ---
integrated_gpu_context = integrated_gpu_queue = integrated_gpu_mmul = None
integrated_gpu_a = integrated_gpu_b = integrated_gpu_c = None

if integrated_gpu_h_A is not None and integrated_gpu_h_C is not None:
    print("\nCreating a integrated_gpu OpenCL context...")
    integrated_gpu_kernel_source = f"""
#define WIDTH {WIDTH}
#define TSM {TSM}
#define TSN {TSN}
#define TSK {TSK}
#define WPTM {WPTM}
#define WPTN {WPTN}
""" + open(integrated_gpu_kernel_name).read()
    integrated_gpu_context = cl.create_some_context()
    integrated_gpu_queue = cl.CommandQueue(integrated_gpu_context)
    integrated_gpu_a = cl.Buffer(integrated_gpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=integrated_gpu_h_A)
    integrated_gpu_b = cl.Buffer(integrated_gpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=integrated_gpu_h_B)
    integrated_gpu_c = cl.Buffer(integrated_gpu_context, cl.mem_flags.WRITE_ONLY, size=integrated_gpu_h_C.nbytes)
    integrated_gpu_program = cl.Program(integrated_gpu_context, integrated_gpu_kernel_source).build()
    integrated_gpu_mmul = integrated_gpu_program.mmul
    integrated_gpu_mmul.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])

# --- Run multiplications ---
print(f"\nStarting {COUNT} OpenCL Matrix Multiplication(s)...")
start_time = time()

for i in range(COUNT):
    try:
        if gpu_mmul is not None and gpu_queue is not None:
            gpu_mmul(
                gpu_queue,
                (GPU_M, N),
                (TS, TS),
                numpy.int32(GPU_M),
                numpy.int32(N),
                numpy.int32(K),
                gpu_a,
                gpu_b,
                gpu_c,
            )

        if integrated_gpu_mmul is not None and integrated_gpu_queue is not None:
            integrated_gpu_mmul(
                integrated_gpu_queue,
                (integrated_gpu_M // WPTM, N // WPTN),
                (TSM // WPTM, TSN // WPTN),
                numpy.int32(integrated_gpu_M),
                numpy.int32(N),
                numpy.int32(K),
                integrated_gpu_a,
                integrated_gpu_b,
                integrated_gpu_c,
            )

        if gpu_queue is not None:
            gpu_queue.flush()
        if integrated_gpu_queue is not None:
            integrated_gpu_queue.flush()

        if gpu_queue is not None:
            gpu_queue.finish()
        if integrated_gpu_queue is not None:
            integrated_gpu_queue.finish()
    except cl.Error as e:
        print(f"OpenCL error: {e}")

run_time = time() - start_time
print(f"End of {COUNT} Matrix Multiplication(s).")
print(f"{run_time:.4f} seconds at {gflop / run_time:.2f} GFLOPS")

# --- Read back results ---
if gpu_queue is not None and gpu_h_C is not None and gpu_c is not None:
    cl.enqueue_copy(gpu_queue, gpu_h_C, gpu_c)

if integrated_gpu_queue is not None and integrated_gpu_h_C is not None and integrated_gpu_c is not None:
    cl.enqueue_copy(integrated_gpu_queue, integrated_gpu_h_C, integrated_gpu_c)

parts = [arr for arr in (gpu_h_C, integrated_gpu_h_C) if arr is not None]
h_C = numpy.concatenate(parts)
sizeC = h_C.size

print("\nRandom samples of h_C:")
for _ in range(10):
    idx = random.randint(0, sizeC - 1)
    print(f"\th_C[{idx}] = {h_C[idx]}")

print(f"\tcval: {CVAL}")
