import os

# Must be set before the first CUDA allocation. The max-batch probe always ran with it; the
# training runs did not, so a batch that fit in a 3-step probe fragmented its way to OOM in
# the real run (run 3, first Colab attempt). Every entry point imports this package first.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
