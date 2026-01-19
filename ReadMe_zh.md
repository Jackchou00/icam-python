# 图像色貌模型（iCAM）的 Python 实现

[English](ReadMe.md) | 简体中文

## 简介

色貌模型作为色度学的延伸，在图像处理领域也有重要的应用，尤其是高动态范围图像的色调映射方面。

iCAM 是 Fairchild 等人提出的一种框架，将色貌模型应用于图像处理。其中 iCAM06 版本用到了当时最先进的色度学方法和模型，比如 CAT02，CAM02，IPT 等。这种框架后来也有改进和衍生，比如 TMOz。

这个项目是 iCAM 的 Python 实现，参考了 iCAM06 的原始代码（MATLAB）。

详细的研究和模型细节请参阅：[iCAM06 调研笔记](https://jackchou00.com/posts/icam06-survey/)


## 参考文献

[1] J. Kuang, G. M. Johnson, and M. D. Fairchild, “iCAM06: A refined image appearance model for HDR image rendering,” Journal of Visual Communication and Image Representation, vol. 18, no. 5, pp. 406–414, Oct. 2007, doi: 10.1016/j.jvcir.2007.06.003.

[2] I. Mehmood, M. Zhou, M. U. Khan, and M. Luo, “CIECAM16-based Tone Mapping of High Dynamic Range Images,” Color and Imaging Conference, vol. 31, pp. 102–107, Nov. 2023, doi: 10.2352/CIC.2023.31.1.20.

## 输入和输出

iCAM 接受 XYZ 图像作为输入，这里的 XYZ 指的是原始绝对三刺激值（tristimulus），未进行色适应（chromatic adaptation）或归一化（normalization）。这些值可以通过相机和对应的颜色校正矩阵（CCM）获取，或通过图像的逆处理获得。

iCAM 的输出图像与输入图像具有相同的尺寸。但输出的 XYZ 值经过压缩处理，可转换为 sRGB 或其他颜色空间，最终在显示器上显示。

## 问题

在原始 iCAM06 代码中发现以下问题：

- 修改后的 CAT02 中 D 参数被错误乘以 0.3。对适应场（adaptation field）与三刺激值在对角矩阵中的关系处理不当（因归一化错误）会导致色适应后图像亮度发生变化。
- CAT02 中 D 的计算公式不正确。应为 $-(L_A+42)$ ，而非 $-(L_A-42)$ 。

## 文件

目前，正在改进项目整体的文件结构，以实现模块化和更好的可读性。

已按照 iCAM 的结构，将仓库分为以下部分：

- `chromatic_adaptation`：色适应
- `colour_space_conversion`：色彩空间转换
- `spatial_process`：空域处理
- `tone_compression`：色调压缩
