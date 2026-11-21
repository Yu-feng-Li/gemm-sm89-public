"""CUDA device selection, context ownership and dispatch identity."""
import os
from contextlib import contextmanager

from cuda.bindings import driver as cuda


def checked(call):
    result, *values = call
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(result)
    return values[0] if values else None


def current_device():
    checked(cuda.cuInit(0))
    context = checked(cuda.cuCtxGetCurrent())
    return checked(cuda.cuCtxGetDevice() if int(context) else cuda.cuDeviceGet(0))


def require_sm89():
    device = current_device()
    attrs = cuda.CUdevice_attribute
    sm = tuple(checked(cuda.cuDeviceGetAttribute(attr, device)) for attr in
               (attrs.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, attrs.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR))
    if sm != (8, 9):
        raise ValueError(f"This kernel requires SM89, found SM{sm[0]}{sm[1]}")
    os.environ["CUTE_DSL_ARCH"] = "sm_89"
    return device


def device_model():
    return checked(cuda.cuDeviceGetName(128, current_device())).split(b"\0", 1)[0].decode()


@contextmanager
def primary_context():
    device = require_sm89()
    context = checked(cuda.cuDevicePrimaryCtxRetain(device))
    previous = checked(cuda.cuCtxGetCurrent())
    checked(cuda.cuCtxSetCurrent(context))
    try:
        yield device
    finally:
        checked(cuda.cuCtxSetCurrent(previous))
        checked(cuda.cuDevicePrimaryCtxRelease(device))
