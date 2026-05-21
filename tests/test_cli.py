"""Test CLI functions."""
import subprocess
from safesched import cli


def test_detect_gpus_no_nvidia(mocker):
    """Test detect_gpus returns empty list when nvidia-smi is not available."""
    # Mock subprocess.run to simulate missing nvidia-smi
    def fake_run(*args, **kwargs):
        return type('obj', (object,), {'returncode': 1, 'stdout': ''})()
    
    mocker.patch('subprocess.run', side_effect=fake_run)
    
    result = cli.detect_gpus()
    assert result == []


def test_detect_gpus_with_nvidia(mocker):
    """Test detect_gpus returns GPU list when nvidia-smi is available."""
    def fake_run(*args, **kwargs):
        return type('obj', (object,), {'returncode': 0, 'stdout': '0\n1\n'})()
    
    mocker.patch('subprocess.run', side_effect=fake_run)
    
    result = cli.detect_gpus()
    assert result == [0, 1]


def test_get_idlest_gpu_no_gpus(mocker):
    """Test get_idlest_gpu returns -1 when no GPUs available."""
    result = cli.get_idlest_gpu([])
    assert result == -1


def test_get_idlest_gpu_single_gpu(mocker):
    """Test get_idlest_gpu with a single GPU."""
    # Mock subprocess to simulate nvidia-smi output: used=500MB, total=2000MB (25% usage)
    def fake_run(*args, **kwargs):
        return type('obj', (object,), {'returncode': 0, 'stdout': '500,2000'})()
    
    mocker.patch('subprocess.run', side_effect=fake_run)
    
    result = cli.get_idlest_gpu([0])
    assert result == 0


def test_get_idlest_gpu_multiple_gpus(mocker):
    """Test get_idlest_gpu selects the least loaded GPU."""
    call_count = [0]
    
    def fake_run(*args, **kwargs):
        # First GPU: 50% usage (1000/2000)
        # Second GPU: 25% usage (500/2000)
        # get_idlest_gpu should select GPU 1
        if call_count[0] == 0:
            call_count[0] += 1
            return type('obj', (object,), {'returncode': 0, 'stdout': '1000,2000'})()
        else:
            return type('obj', (object,), {'returncode': 0, 'stdout': '500,2000'})()
    
    mocker.patch('subprocess.run', side_effect=fake_run)
    
    result = cli.get_idlest_gpu([0, 1])
    assert result == 1
