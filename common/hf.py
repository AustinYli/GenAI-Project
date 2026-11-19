"""Small helpers so the code works on both transformers 4.x and 5.x."""


def dtype_kwargs(torch_dtype):
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) >= (4, 56):
        return {"dtype": torch_dtype}
    return {"torch_dtype": torch_dtype}
