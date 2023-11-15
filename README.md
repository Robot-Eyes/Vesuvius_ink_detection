# Vesuvius_ink_detection

This repo contains my solution to [Kaggle Vesuvius competition](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection). This method can achieve 0.647 Private LB which is in the golden medal zone.

![image](https://github.com/Robot-Eyes/Vesuvius_ink_detection/assets/100538999/f5657ce3-c448-408e-aeb5-e40022d73ca7)


## Problem statement

In 79 AD, Mount Vesuvius erupted, burying a library of ancient Roman papyrus scrolls in hot mud and ash. Nearly 2,000 years later, scientists are using 3D X-ray scans with 4µm resolution to detect ink on carbonized scrolls, hoping to recover lost Greek literature from the ashes. But as the ink does not show up readily in X-ray scans, machine learning techniques that can distinguish nuance between ink and non-ink voxels are needed.

## Method

The 3D scans are represented as a grid of voxels, each voxel having a rectangular shape with dimensions DxWxH. The values of these voxels represent the density of the scanned items. Many winning solutions apply 3D segmentation models directly on these scans to extract regions with ink. 

However, essentially these scrolls are not flat, and their shapes are irregular. To address this, an extra step is introduced in my solution. I reconstructed the 3D structure of the papyrus scrolls from the voxel values, and then flattened them into regular rectangular shapes. 

By implementing this additional step, the method achieves significant improvements in performance. Specifically, two UNet-3D models ensemble could reach performance of "golden medal zone". Furthermore, it opens the door for even more advanced models like Segformer to be applied and potentially enhanced further.
