"""Test ResourceMonitor class."""
import time
from safesched.monitor import ResourceMonitor


def test_monitor_init():
    """Test ResourceMonitor initialization."""
    monitor = ResourceMonitor(gpu_ids=[])
    assert monitor.gpu_ids == []
    assert monitor._stop is False
    monitor.shutdown()


def test_monitor_shutdown():
    """Test ResourceMonitor shutdown."""
    monitor = ResourceMonitor(gpu_ids=[])
    monitor.shutdown()
    assert monitor._stop is True


def test_monitor_get_summary(mocker):
    """Test ResourceMonitor.get_summary() with mocked psutil."""
    # Mock psutil to return predictable values
    mocker.patch('psutil.cpu_percent', return_value=50.0)
    mocker.patch('psutil.virtual_memory', return_value=type('obj', (object,), {'percent': 60.0})())
    mocker.patch('psutil.disk_usage', return_value=type('obj', (object,), {'percent': 30.0})())
    
    monitor = ResourceMonitor(gpu_ids=[])
    time.sleep(0.1)  # Wait for monitor thread to update
    
    summary = monitor.get_summary()
    assert 'CPU=' in summary
    assert 'MEM=' in summary
    assert 'IO=' in summary
    
    monitor.shutdown()


def test_monitor_is_overloaded_normal(mocker):
    """Test is_overloaded returns False when resources are normal."""
    mocker.patch('psutil.cpu_percent', return_value=30.0)
    mocker.patch('psutil.virtual_memory', return_value=type('obj', (object,), {'percent': 40.0})())
    mocker.patch('psutil.disk_usage', return_value=type('obj', (object,), {'percent': 20.0})())
    
    monitor = ResourceMonitor(gpu_ids=[])
    time.sleep(0.1)
    
    assert monitor.is_overloaded() is False
    monitor.shutdown()


def test_monitor_is_overloaded_high_cpu(mocker):
    """Test is_overloaded returns True when CPU is high."""
    mocker.patch('psutil.cpu_percent', return_value=95.0)
    mocker.patch('psutil.virtual_memory', return_value=type('obj', (object,), {'percent': 40.0})())
    mocker.patch('psutil.disk_usage', return_value=type('obj', (object,), {'percent': 20.0})())
    
    monitor = ResourceMonitor(gpu_ids=[])
    time.sleep(0.1)
    
    assert monitor.is_overloaded() is True
    monitor.shutdown()
