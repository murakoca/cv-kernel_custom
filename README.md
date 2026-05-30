GPU Compute Kernels Library

Overview

This repository provides high-performance GPU implementations of commonly used computational kernels for image processing and string similarity tasks.

The goal is to create a framework-agnostic acceleration layer that can be integrated into any computer vision, OCR, or data processing pipeline.

The current scope includes:

* Contrast enhancement kernel (CLAHE)
* Batch string distance kernel (Levenshtein)

All implementations target CUDA (primary) with potential Metal / Vulkan compute extensions in future.

⸻

1. CLAHE Kernel (Contrast Limited Adaptive Histogram Equalization)

Problem Statement

Adaptive histogram equalization is widely used in image preprocessing pipelines to enhance contrast under non-uniform lighting conditions.

Standard CPU implementations (e.g. OpenCV cv2.createCLAHE) are:

* High quality
* But computationally expensive
* Not suitable for real-time or high-throughput pipelines

This becomes a bottleneck in systems processing large image batches or video streams.

⸻

Solution: GPU-Accelerated CLAHE Kernel

A CUDA-based implementation of CLAHE designed for massively parallel execution while preserving visual fidelity.

Expected acceleration: 10–20× vs CPU implementations

⸻

Algorithm Overview

1. Image Tiling

Input image is partitioned into uniform tiles (e.g. 8×8 or 16×16).

2. Histogram Computation

Each tile computes its histogram independently in parallel.

3. Clipping

Histogram values are clipped using a configurable threshold to suppress noise amplification.

4. Local Equalization

Each tile is equalized independently based on its clipped histogram.

5. Interpolation

Adjacent tiles are blended using bilinear interpolation to avoid boundary artifacts.

6. GPU Parallelism Model

Each tile is mapped to a CUDA thread block, enabling full spatial parallelization.

⸻

Benefits

* High-speed image contrast enhancement
* Suitable for real-time pipelines
* Stable output quality comparable to OpenCV CLAHE
* Scales with image resolution