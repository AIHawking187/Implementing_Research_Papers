# Implementing Research Papers
## 1. Normalizing Flows
Normalizing flows are a class of generative models that learn complex probability distributions 
by transforming a simple base distribution through a sequence of invertible mappings.

### Key Concepts

#### 1. Change of Variables Formula
The fundamental principle behind normalizing flows is the change of variables formula:
p_x(x) = p_z(z) * |det(dz/dx)|
where:
- p_x(x) is the target distribution
- p_z(z) is the base distribution (usually standard normal)
- |det(dz/dx)| is the absolute determinant of the Jacobian

#### 2. Flow Types
Different flow architectures provide different ways to transform the data while maintaining
invertibility and tractable Jacobian determinants:
- Planar: Uses planar transformations (f(z) = z + u*h(w^T*z + b))
- Radial: Uses radial transformations around a learned reference point
- RealNVP: Uses affine coupling layers with masked transformations
- MAF: Uses autoregressive transformations for increased flexibility
"""
