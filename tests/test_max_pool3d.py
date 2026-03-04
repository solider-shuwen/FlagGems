"""
Test suite for max_pool3d operator.

This test module validates the correctness, precision, and performance
of the max_pool3d operator implementation following FlagGems testing conventions.

Test coverage description:
- Input sizes: small (4×4×4), regular (32×32×32), large (128×128×128)
- Input dimensions: 4D (C,D,H,W), 5D (N,C,D,H,W)
- Data types: float16, float32, bfloat16
- Parameter patterns: kernel_size, stride, padding, dilation, ceil_mode, return_indices
- Functional completeness: basic pooling, index return, 4D/5D input, batch processing
"""

import pytest
import torch

from flag_gems.ops import max_pool3d

# ============================================================================
# Test data definitions (following competition requirements)
# ============================================================================

# Input size coverage (competition requirement: small, regular, large sizes)
POOL3D_SHAPES = [
    # Small sizes
    (1, 1, 4, 4, 4),  # N=1, C=1, D=4, H=4, W=4
    (2, 3, 8, 8, 8),  # N=2, C=3, D=8, H=8, W=8
    # Regular sizes
    (2, 16, 32, 32, 32),  # N=2, C=16, D=32, H=32, W=32
    (4, 64, 64, 64, 64),  # N=4, C=64, D=64, H=64, W=64
    # Large sizes
    (2, 128, 128, 128, 128),  # N=2, C=128, D=128, H=128, W=128
    (4, 256, 256, 256, 256),  # N=4, C=256, D=256, H=256, W=256
]

# Data type coverage (competition requirement: at least support float32/float16)
FLOAT_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
]

# Precision standards (competition requirement standards)
# rtol = 1e-4 (all floating point types)
# atol varies by data type
ATOL_DICT = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}


# ============================================================================
# Helper functions
# ============================================================================


def assert_close(actual, expected, rtol=1e-4, atol=None, dtype=torch.float32):
    """
    Verify precision using torch.allclose (competition requirement standards)

    Args:
        actual: FlagGems implementation result
        expected: PyTorch reference result
        rtol: relative error tolerance (default 1e-4)
        atol: absolute error tolerance (based on data type)
        dtype: data type
    """
    if atol is None:
        atol = ATOL_DICT.get(dtype, 1e-5)

    # Compare using torch.allclose (competition standard)
    assert torch.allclose(
        actual, expected, rtol=rtol, atol=atol, equal_nan=True
    ), f"Results don't match: max diff={(actual - expected).abs().max().item()}"


def create_tensor(shape, dtype, device="cuda"):
    """Create test tensor"""
    x = torch.randn(shape, dtype=dtype, device=device)
    return x


# ============================================================================
# 1. Input size coverage tests (competition requirement: small, regular, large categories)
# ============================================================================


class TestMaxPool3DInputSize:
    """Test input size coverage"""

    @pytest.mark.max_pool3d
    def test_size_very_small(self):
        """Test: extremely small size (1, 1, 4, 4, 4)"""
        x = create_tensor((1, 1, 4, 4, 4), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_size_small(self):
        """Test: small size (2, 3, 8, 8, 8)"""
        x = create_tensor((2, 3, 8, 8, 8), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_size_medium_32(self):
        """Test: regular size (2, 16, 32, 32, 32)"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_size_medium_64(self):
        """Test: regular size (4, 64, 64, 64, 64)"""
        x = create_tensor((4, 64, 64, 64, 64), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_size_large_128(self):
        """Test: large size (2, 128, 128, 128, 128)"""
        x = create_tensor((2, 128, 128, 128, 128), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_size_large_256(self):
        """Test: large size (2, 128, 64, 64, 64)"""
        x = create_tensor((2, 128, 64, 64, 64), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 2. Input dimension coverage tests (competition requirement: cover all valid dimensions)
# ============================================================================


class TestMaxPool3DInputDimensions:
    """Test input dimension coverage: 4D (C,D,H,W), 5D (N,C,D,H,W)"""

    @pytest.mark.max_pool3d
    def test_dim_4d_small(self):
        """Test: 4D tensor - small size (C, D, H, W)"""
        x = create_tensor((1, 4, 4, 4), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)
        # Verify output is also 4D
        assert y_gems.dim() == 4

    @pytest.mark.max_pool3d
    def test_dim_4d_medium(self):
        """Test: 4D tensor - medium size (C, D, H, W)"""
        x = create_tensor((16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)
        # Verify output is also 4D
        assert y_gems.dim() == 4

    @pytest.mark.max_pool3d
    def test_dim_5d_small(self):
        """Test: 5D tensor - small size (N, C, D, H, W)"""
        x = create_tensor((2, 3, 8, 8, 8), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)
        # Verify output is also 5D
        assert y_gems.dim() == 5

    @pytest.mark.max_pool3d
    def test_dim_5d_large(self):
        """Test: 5D tensor - large size (N, C, D, H, W)"""
        x = create_tensor((4, 64, 64, 64, 64), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)
        # Verify output is also 5D
        assert y_gems.dim() == 5


# ============================================================================
# 3. Data type coverage tests (competition requirement: at least support float32/float16)
# ============================================================================


class TestMaxPool3DDataTypes:
    """Test data type coverage: float16, float32, bfloat16"""

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_float_dtypes(self, dtype):
        """Test: all floating point data types"""
        x = create_tensor((2, 16, 32, 32, 32), dtype)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)


# ============================================================================
# 4. Parameter pattern coverage tests (competition requirement: default values, boundary values, special values)
# ============================================================================


class TestMaxPool3DParameterPatterns:
    """Test parameter patterns: kernel_size, stride, padding, dilation, ceil_mode"""

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("kernel_size", [1, 2, 3, 4])
    def test_various_kernel_sizes(self, kernel_size):
        """Test: various kernel_size values"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=kernel_size)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=kernel_size)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("stride", [1, 2, 3])
    def test_various_strides(self, stride):
        """Test: various stride values"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3, stride=stride)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3, stride=stride)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("padding", [0, 1, 3])
    def test_various_padding(self, padding):
        """Test: various padding values"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=7, padding=padding)
        # y_gems = torch.nn.functional.max_pool3d(x, kernel_size=7, padding=padding)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=7, padding=padding)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_various_dilation(self, dilation):
        """Test: various dilation values"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3, dilation=dilation)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3, dilation=dilation)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("ceil_mode", [False, True])
    def test_ceil_mode(self, ceil_mode):
        """Test: ceil_mode parameter"""
        x = create_tensor((2, 16, 31, 31, 31), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3, stride=2, ceil_mode=ceil_mode)
        y_torch = torch.nn.functional.max_pool3d(
            x, kernel_size=3, stride=2, ceil_mode=ceil_mode
        )
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_tuple_kernel_size(self):
        """Test: tuple kernel_size parameter"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=(2, 3, 4))
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=(2, 3, 4))
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_tuple_stride(self):
        """Test: tuple stride parameter"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3, stride=(1, 2, 1))
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3, stride=(1, 2, 1))
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_tuple_padding(self):
        """Test: tuple padding parameter"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3, padding=(0, 1, 0))
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3, padding=(0, 1, 0))
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_tuple_dilation(self):
        """Test: tuple dilation parameter"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3, dilation=(1, 2, 1))
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3, dilation=(1, 2, 1))
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_default_stride(self):
        """Test: default stride (equals kernel_size)"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=3)  # stride defaults to kernel_size
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)


# ============================================================================
# 5. Functional completeness tests (competition requirement: all functional branches)
# ============================================================================


class TestMaxPool3DFunctionalCompleteness:
    """Test functional completeness: basic pooling, index return, 4D/5D input, batch processing"""

    @pytest.mark.max_pool3d
    def test_basic_pooling(self):
        """Test: basic 3D pooling operation"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_return_indices_false(self):
        """Test: return_indices=False (return only output)"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems = max_pool3d(x, kernel_size=2, return_indices=False)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2, return_indices=False)
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)
        # Verify single tensor is returned
        assert isinstance(y_gems, torch.Tensor)

    @pytest.mark.max_pool3d
    def test_return_indices_true(self):
        """Test: return_indices=True (return output and indices)"""
        x = create_tensor((2, 16, 32, 32, 32), torch.float32)
        y_gems, idx_gems = max_pool3d(x, kernel_size=2, return_indices=True)
        y_torch, idx_torch = torch.nn.functional.max_pool3d(
            x, kernel_size=2, return_indices=True
        )
        # Verify output values
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)
        # Verify indices (integers should match exactly)
        assert torch.equal(idx_gems, idx_torch)
        # Verify tuple is returned
        assert isinstance(y_gems, torch.Tensor)
        assert isinstance(idx_gems, torch.Tensor)

    @pytest.mark.max_pool3d
    def test_return_indices_4d_input(self):
        """Test: 4D input with return_indices=True"""
        x = create_tensor((16, 32, 32, 32), torch.float32)
        y_gems, idx_gems = max_pool3d(x, kernel_size=2, return_indices=True)
        y_torch, idx_torch = torch.nn.functional.max_pool3d(
            x, kernel_size=2, return_indices=True
        )
        # Verify output values
        assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)
        # Verify indices
        assert torch.equal(idx_gems, idx_torch)
        # Verify output dimensions
        assert y_gems.dim() == 4
        assert idx_gems.dim() == 4

    @pytest.mark.max_pool3d
    def test_batch_processing(self):
        """Test: batch processing (multiple tensors)"""
        tensors = [
            create_tensor((2, 16, 32, 32, 32), torch.float32),
            create_tensor((4, 32, 64, 64, 64), torch.float32),
            create_tensor((1, 8, 16, 16, 16), torch.float32),
        ]
        for x in tensors:
            y_gems = max_pool3d(x, kernel_size=2)
            y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
            assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_consistency_with_pytorch(self):
        """Test: consistency with PyTorch implementation"""
        # Test multiple times with random tensors
        for _ in range(5):
            x = create_tensor((2, 16, 32, 32, 32), torch.float32)
            y_gems = max_pool3d(x, kernel_size=2)
            y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
            assert_close(y_gems, y_torch, rtol=1e-4, atol=1.3e-6, dtype=torch.float32)

    @pytest.mark.max_pool3d
    def test_empty_tensor(self):
        """Test: empty tensor handling"""
        x = torch.randn(0, 16, 32, 32, 32, dtype=torch.float32, device="cuda")
        y_gems = max_pool3d(x, kernel_size=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=2)
        assert y_gems.shape == y_torch.shape
        assert y_gems.numel() == 0


# ============================================================================
# 6. Comprehensive coverage tests (parameterized combinations)
# ============================================================================


class TestMaxPool3DComprehensive:
    """Comprehensive tests: multi-dimensional combination coverage"""

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize(
        "shape,kernel_size",
        [
            ((1, 1, 4, 4, 4), 2),
            ((2, 3, 8, 8, 8), 3),
            ((2, 16, 32, 32, 32), 2),
            ((4, 64, 64, 64, 64), 3),
        ],
    )
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_shape_dtype_combination(self, shape, kernel_size, dtype):
        """Test: combination of shape and data type"""
        x = create_tensor(shape, dtype)
        y_gems = max_pool3d(x, kernel_size=kernel_size)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=kernel_size)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize(
        "shape,kernel_size,stride,padding",
        [
            ((1, 1, 4, 4, 4), 2, 1, 0),
            ((2, 3, 8, 8, 8), 3, 2, 1),
            ((2, 16, 32, 32, 32), 2, 2, 1),
            ((4, 64, 64, 64, 64), 3, 3, 1),
        ],
    )
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_parameter_combinations(self, shape, kernel_size, stride, padding, dtype):
        """Test: parameter combinations"""
        x = create_tensor(shape, dtype)
        y_gems = max_pool3d(x, kernel_size=kernel_size, stride=stride, padding=padding)
        y_torch = torch.nn.functional.max_pool3d(
            x, kernel_size=kernel_size, stride=stride, padding=padding
        )
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize(
        "shape,dtype,kernel_size",
        [
            ((1, 1, 4, 4, 4), torch.float16, 2),
            ((2, 3, 8, 8, 8), torch.float32, 3),
            ((2, 16, 32, 32, 32), torch.bfloat16, 2),
            ((4, 64, 64, 64, 64), torch.float16, 3),
        ],
    )
    def test_typical_use_cases(self, shape, dtype, kernel_size):
        """Test: typical use cases"""
        x = create_tensor(shape, dtype)
        y_gems = max_pool3d(x, kernel_size=kernel_size)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=kernel_size)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize(
        "shape,kernel_size,return_indices",
        [
            ((1, 1, 4, 4, 4), 2, False),
            ((2, 3, 8, 8, 8), 3, True),
            ((2, 16, 32, 32, 32), 2, False),
            ((4, 64, 64, 64, 64), 3, True),
        ],
    )
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_return_indices_combinations(
        self, shape, kernel_size, return_indices, dtype
    ):
        """Test: return_indices parameter combinations"""
        x = create_tensor(shape, dtype)
        result_gems = max_pool3d(
            x, kernel_size=kernel_size, return_indices=return_indices
        )
        result_torch = torch.nn.functional.max_pool3d(
            x, kernel_size=kernel_size, return_indices=return_indices
        )

        if return_indices:
            y_gems, idx_gems = result_gems
            y_torch, idx_torch = result_torch
            atol = ATOL_DICT[dtype]
            assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)
            assert torch.equal(idx_gems, idx_torch)
        else:
            atol = ATOL_DICT[dtype]
            assert_close(result_gems, result_torch, rtol=1e-4, atol=atol, dtype=dtype)


# ============================================================================
# 7. 特殊场景测试
# ============================================================================


class TestMaxPool3DSpecialCases:
    """Test special scenarios"""

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_kernel_size_1(self, dtype):
        """Test: kernel_size=1 (identity operation)"""
        x = create_tensor((2, 16, 32, 32, 32), dtype)
        y_gems = max_pool3d(x, kernel_size=1)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=1)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)
        # When kernel_size=1, output should equal input
        assert_close(y_gems, x, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_large_kernel(self, dtype):
        """Test: large kernel (close to input size)"""
        x = create_tensor((2, 16, 16, 16, 16), dtype)
        y_gems = max_pool3d(x, kernel_size=8, padding=4)
        # y_gems = torch.nn.functional.max_pool3d(x, kernel_size=8, padding=4)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=8, padding=4)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_non_uniform_kernel(self, dtype):
        """Test: non-uniform kernel (2, 3, 4)"""
        x = create_tensor((2, 16, 32, 32, 32), dtype)
        y_gems = max_pool3d(x, kernel_size=(2, 3, 4))
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=(2, 3, 4))
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_asymmetric_stride_padding(self, dtype):
        """Test: asymmetric stride and padding"""
        x = create_tensor((2, 16, 32, 32, 32), dtype)
        y_gems = max_pool3d(x, kernel_size=3, stride=(1, 2, 1), padding=(0, 1, 0))
        y_torch = torch.nn.functional.max_pool3d(
            x, kernel_size=3, stride=(1, 2, 1), padding=(0, 1, 0)
        )
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)

    @pytest.mark.max_pool3d
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dilation_greater_than_1(self, dtype):
        """Test: dilation > 1 (dilated pooling)"""
        x = create_tensor((2, 16, 32, 32, 32), dtype)
        y_gems = max_pool3d(x, kernel_size=3, dilation=2)
        y_torch = torch.nn.functional.max_pool3d(x, kernel_size=3, dilation=2)
        atol = ATOL_DICT[dtype]
        assert_close(y_gems, y_torch, rtol=1e-4, atol=atol, dtype=dtype)
