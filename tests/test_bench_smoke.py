def test_bench_imports():
    import bench
    from bench.errors import BenchError
    assert issubclass(BenchError, Exception)
    assert bench.__version__
