"""Single-machine numerical kernels.

Pure pandas/NumPy implementations of the per-user algorithms. The Spark
pipeline distributes users across executors and applies these kernels to
each partition/group; they can also be used directly on small datasets.
"""
