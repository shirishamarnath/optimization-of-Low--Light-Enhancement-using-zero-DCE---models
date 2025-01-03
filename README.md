# Optimization of Low-Light Enhancement using Zero-DCE++ Models

## Overview
This repository contains the implementation and optimization of the Zero-DCE++ model for low-light image enhancement. By leveraging *8-bit quantization*, the model is optimized for deployment on resource-constrained devices like mobile phones and embedded systems, achieving enhanced image quality with reduced computational overhead.

### Key Features:
- *Model Optimization*: Reduction of model precision to 8-bit integer values for better computational efficiency.
- *Low-Light Image Enhancement*: Real-time enhancement with minimal artifacts.
- *Deployment-Ready*: Adapted for mobile and embedded systems using TensorFlow Lite.
- *Evaluation Metrics*: Includes SSIM, PSNR, and inference time for quantitative analysis.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Results](#results)
4. [Future Work](#future-work)
5. [References](#references)

---

## Introduction
Low-light image enhancement is crucial in domains like surveillance, autonomous driving, and photography. Traditional methods often fail to balance brightness, contrast, and detail. Deep learning models like Zero-DCE++ provide a robust alternative by estimating correction curves directly from low-light images without paired data. 

This project focuses on optimizing Zero-DCE++ for real-time applications by employing *8-bit quantization*, significantly reducing its computational footprint while maintaining high image quality.

---

## Methodology
### Zero-DCE++ Model
- The Zero-DCE++ model estimates pixel correction curves using deep convolutional networks.
- Enhancements include better curve estimation and improved color fidelity over the original Zero-DCE.

### 8-Bit Quantization
- *Post-Training Quantization*: Reduces model size by converting weights to 8-bit precision after training.
- *Quantization-Aware Training*: Incorporates 8-bit precision during training for minimal accuracy loss.

### Training and Evaluation
- The model was trained and tested using the LOL (Low-Light) dataset.
- Evaluation metrics include:
  - *Structural Similarity Index (SSIM)*
  - *Peak Signal-to-Noise Ratio (PSNR)*
  - *Inference Time*

---

## Results
### Quantitative Metrics
- *PSNR*: Achieved high peak signal-to-noise ratios in enhanced images.
- *SSIM*: Maintained structural similarity between enhanced and original images.

### Visual Results
Before and after comparison:
- Low-light images enhanced with improved brightness, contrast, and color fidelity.

### Real-Time Performance
- Achieved significant reduction in inference time.
- Successfully deployed on mobile and embedded systems with low computational power.

---

## Future Work
1. Address minor color inaccuracies in extremely dark conditions.
2. Explore hybrid techniques combining supervised learning with zero-reference methods.
3. Optimize for specific hardware architectures like FPGAs or custom ASICs.
4. Implement advanced quantization techniques such as mixed-precision quantization.

---

## References
1. Guo, X., Li, Y., & Ling, H. (2020). "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement."
2. Chen, W., Xie, C., & Lin, Y. (2021). "Zero-DCE++: Low-Light Image Enhancement Without Paired Data."
3. Wang, S., Wang, X., Liang, J., & Fan, X. (2021). "Optimized Low-Light Image Enhancement Using a Combination of Convolutional Neural Networks and Image Processing Techniques."
4. Tan, Y., Ma, K., Zhang, X., & Wang, H. (2021). "Comparative Analysis of Deep Learning-Based Low-Light Image Enhancement Models on Mobile Platforms."

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
