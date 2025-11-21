# monitoring/metrics.py
"""
Metrics collection and health monitoring
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

from utils.performance import performance_monitor
from database import DatabaseManager

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collect and store system metrics"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.collection_name = "metrics"
        self.running = False
        self.collection_task = None
    
    async def start_collection(self, interval: int = 60):
        """Start periodic metrics collection"""
        self.running = True
        self.collection_task = asyncio.create_task(self._collect_metrics(interval))
        logger.info(f"Started metrics collection every {interval} seconds")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
        logger.info("Stopped metrics collection")
    
    async def _collect_metrics(self, interval: int):
        """Collect metrics periodically"""
        while self.running:
            try:
                await self._collect_current_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_current_metrics(self):
        """Collect current system metrics"""
        timestamp = datetime.utcnow()
        
        # Get performance statistics
        stats = performance_monitor.get_statistics()
        
        # Collect system metrics
        system_metrics = {
            'timestamp': timestamp,
            'type': 'system_metrics',
            'data': {
                'performance_stats': stats,
                'collection_interval': 60
            }
        }
        
        # Store in database
        try:
            await self.db.db[self.collection_name].insert_one(system_metrics)
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    async def get_metrics_summary(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get metrics summary for time range"""
        try:
            cursor = self.db.db[self.collection_name].find({
                'timestamp': {
                    '$gte': start_time,
                    '$lte': end_time
                }
            })
            
            metrics = await cursor.to_list(length=None)
            
            return {
                'start_time': start_time,
                'end_time': end_time,
                'total_metrics': len(metrics),
                'metrics': metrics
            }
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}

class HealthChecker:
    """Check system health"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            'timestamp': datetime.utcnow(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        try:
            # Check database connectivity
            health_status['checks']['database'] = await self._check_database()
            
            # Check performance metrics
            health_status['checks']['performance'] = await self._check_performance()
            
            # Check system resources
            health_status['checks']['resources'] = await self._check_resources()
            
            # Determine overall status
            if any(check['status'] == 'unhealthy' for check in health_status['checks'].values()):
                health_status['overall_status'] = 'unhealthy'
            elif any(check['status'] == 'degraded' for check in health_status['checks'].values()):
                health_status['overall_status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'timestamp': datetime.utcnow(),
                'overall_status': 'unhealthy',
                'error': str(e)
            }
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            # Test database connection
            await self.db.db.admin.command('ping')
            
            # Check database stats
            stats = await self.db.db.command('dbStats')
            
            return {
                'status': 'healthy',
                'response_time': 0,  # Would measure actual ping time
                'database_size': stats.get('dataSize', 0),
                'collections': stats.get('collections', 0)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def _check_performance(self) -> Dict[str, Any]:
        """Check performance metrics"""
        try:
            stats = performance_monitor.get_statistics()
            
            # Check for performance issues
            issues = []
            
            # Check if we have recent metrics
            if stats['total_metrics'] == 0:
                issues.append("No metrics collected")
            
            # Check timing percentiles if available
            if 'timing_percentiles' in stats:
                p95 = stats['timing_percentiles'].get('p95', 0)
                if p95 > 5.0:  # More than 5 seconds
                    issues.append(f"High p95 latency: {p95:.2f}s")
            
            if issues:
                return {
                    'status': 'degraded',
                    'issues': issues,
                    'stats': stats
                }
            else:
                return {
                    'status': 'healthy',
                    'stats': stats
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def _check_resources(self) -> Dict[str, Any]:
        """Check system resources"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            issues = []
            
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent}%")
            
            if disk.percent > 90:
                issues.append(f"High disk usage: {disk.percent}%")
            
            if issues:
                return {
                    'status': 'degraded',
                    'issues': issues,
                    'metrics': {
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'disk_percent': disk.percent
                    }
                }
            else:
                return {
                    'status': 'healthy',
                    'metrics': {
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'disk_percent': disk.percent
                    }
                }
        except ImportError:
            return {
                'status': 'degraded',
                'issues': ['psutil not available']
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

# Global instances
metrics_collector = None
health_checker = None

def initialize_monitoring(db: DatabaseManager):
    """Initialize monitoring system"""
    global metrics_collector, health_checker
    
    metrics_collector = MetricsCollector(db)
    health_checker = HealthChecker(db)
    
    # Start metrics collection
    asyncio.create_task(metrics_collector.start_collection(interval=60))
    
    logger.info("Monitoring system initialized")