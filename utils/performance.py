# utils/performance.py
"""
Performance monitoring and metrics collection
Inspired by enterprise-grade monitoring patterns
"""

import time
import functools
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data"""
    name: str
    value: float
    unit: str
    timestamp: float
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None

class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: deque = deque(maxlen=max_history)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        self.metrics.append(metric)
        logger.debug(f"Recorded metric: {metric.name} = {metric.value} {metric.unit}")
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter"""
        self.counters[name] += value
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit='count',
            timestamp=time.time(),
            tags=tags or {}
        )
        self.record_metric(metric)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge value"""
        self.gauges[name] = value
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit='gauge',
            timestamp=time.time(),
            tags=tags or {}
        )
        self.record_metric(metric)
    
    def start_timer(self, name: str):
        """Start a timer"""
        self.start_times[name] = time.time()
    
    def stop_timer(self, name: str, tags: Dict[str, str] = None):
        """Stop a timer and record duration"""
        if name not in self.start_times:
            logger.warning(f"Timer '{name}' was not started")
            return
        
        duration = time.time() - self.start_times[name]
        metric = PerformanceMetric(
            name=name,
            value=duration,
            unit='seconds',
            timestamp=time.time(),
            tags=tags or {}
        )
        self.record_metric(metric)
        del self.start_times[name]
        return duration
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'total_metrics': len(self.metrics),
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'active_timers': list(self.start_times.keys()),
            'recent_metrics': [asdict(m) for m in list(self.metrics)[-10:]]
        }
        
        # Calculate percentiles for recent timings
        recent_timings = [m.value for m in self.metrics if m.unit == 'seconds']
        if recent_timings:
            recent_timings.sort()
            stats['timing_percentiles'] = {
                'p50': recent_timings[len(recent_timings) // 2],
                'p90': recent_timings[int(len(recent_timings) * 0.9)],
                'p95': recent_timings[int(len(recent_timings) * 0.95)],
                'p99': recent_timings[int(len(recent_timings) * 0.99)]
            }
        
        return stats

# Global performance monitor
performance_monitor = PerformanceMonitor()

def monitor_performance(name: str, tags: Dict[str, str] = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            performance_monitor.start_timer(name)
            try:
                result = await func(*args, **kwargs)
                performance_monitor.increment_counter(f"{name}_success", tags=tags)
                return result
            except Exception as e:
                performance_monitor.increment_counter(f"{name}_error", tags=tags)
                raise
            finally:
                performance_monitor.stop_timer(name, tags=tags)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            performance_monitor.start_timer(name)
            try:
                result = func(*args, **kwargs)
                performance_monitor.increment_counter(f"{name}_success", tags=tags)
                return result
            except Exception as e:
                performance_monitor.increment_counter(f"{name}_error", tags=tags)
                raise
            finally:
                performance_monitor.stop_timer(name, tags=tags)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def track_memory_usage():
    """Track memory usage"""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        performance_monitor.set_gauge('memory_rss', memory_info.rss / 1024 / 1024, {'unit': 'MB'})
        performance_monitor.set_gauge('memory_vms', memory_info.vms / 1024 / 1024, {'unit': 'MB'})
        
    except ImportError:
        logger.warning("psutil not available for memory tracking")

def track_cpu_usage():
    """Track CPU usage"""
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        performance_monitor.set_gauge('cpu_usage', cpu_percent, {'unit': 'percent'})
        
    except ImportError:
        logger.warning("psutil not available for CPU tracking")

class AsyncLimiter:
    """Rate limiter for async operations"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks = 0
    
    async def acquire(self):
        """Acquire a slot"""
        async with self.semaphore:
            self.active_tasks += 1
            performance_monitor.increment_counter('limiter_acquired')
    
    async def release(self):
        """Release a slot"""
        self.active_tasks -= 1
        performance_monitor.increment_counter('limiter_released')
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()