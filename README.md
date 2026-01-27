# CoGeoAD
CoGeoAD: Hierarchical Color-Geometric Fusion with Multi-View Attention for Zero-Shot 3D Anomaly Detection
## Introduction 
 Zero-shot 3D anomaly detection is essential for industrial quality inspection, where labeled anomaly samples are scarce. Meanwhile, existing methods lack an effective mechanism to fuse complementary 2D color cues with 3D geometric structures, limiting their ability to detect both surface and structural defects in a unified framework. To address these issues, we propose CoGeoAD, a unified zero-shot framework that bridges the 2D-3D domain gap by constructing pixel-aligned paired colored and geometric point clouds. The framework introduces a Data-Driven Multi-View Attention (MVA) mechanism to adaptively aggregate 3D features and a Multi-Stage Color-Geometric Fusion (MS-CGF) module to hierarchically integrate multi-scale features from both modalities. Extensive experiments on the MVTec3D-AD and Eyecandies benchmarks demonstrate that CoGeoAD achieves state-of-the-art performance, effectively capturing both structural and textural anomalies in complex industrial scenarios.
## Motivation
<p align="center">
  <a href="assets/motivation-1.pdf">
    <img src="assets/motivation-1.png" width="56%" alt="Click to view PDF"/>
  </a>
  <a href="assets/motivation-2.pdf">
    <img src="assets/motivation-2.png" width="42%" alt="Click to view PDF"/>
  </a>
</p>
## Overview of PointAD
![overview](./assets/overview.png)