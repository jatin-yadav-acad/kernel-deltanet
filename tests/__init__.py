"""
Tests module for Kernel-DeltaNet
"""

def get_master_test_suite():
    from tests.test_all_mlsys import run_master_test_suite
    return run_master_test_suite

__all__ = ["get_master_test_suite"]
