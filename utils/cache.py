# utils/cache.py
"""
Response Caching System
Intelligent caching of LLM responses for performance optimization
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import diskcache as dc
except ImportError:
    dc = None
    logger.warning("diskcache not installed. Cache will use memory-based storage only.")

class CacheStrategy(Enum):
    """Cache strategies for different scenarios"""
    EXACT_MATCH = "exact_match"           # Exact hash match
    SEMANTIC_SIMILARITY = "semantic_similarity"  # Semantic similarity
    PARTIAL_MATCH = "partial_match"       # Partial content match
    FUZZY_MATCH = "fuzzy_match"          # Fuzzy string matching


class CacheEvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    SIZE = "size"                  # Cache size limit
    MEMORY = "memory"              # Memory-based eviction


@dataclass
class CacheConfig:
    """Configuration for response caching"""
    # Storage settings
    cache_dir: str = "/tmp/agent_ui_cache"
    max_cache_size_mb: int = 500
    max_entries: int = 10000
    
    # TTL settings
    default_ttl_seconds: int = 3600  # 1 hour
    max_ttl_seconds: int = 86400     # 24 hours
    
    # Eviction policy
    eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU
    
    # Similarity settings
    enable_similarity_cache: bool = True
    similarity_threshold: float = 0.85
    max_similarity_cache_size: int = 1000
    
    # Performance settings
    enable_compression: bool = True
    enable_stats: bool = True
    batch_processing: bool = True
    
    # Memory management
    max_memory_usage_mb: int = 200
    gc_interval_seconds: int = 300  # 5 minutes


@dataclass
class CacheEntry:
    """Cache entry metadata"""
    key: str
    hash_key: str
    created_at: float
    last_accessed: float
    access_count: int
    ttl_expires: float
    size_bytes: int
    content_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, current_time: float = None) -> bool:
        """Check if cache entry is expired"""
        if current_time is None:
            current_time = time.time()
        return current_time > self.ttl_expires
    
    def access(self):
        """Update access statistics"""
        self.last_accessed = time.time()
        self.access_count += 1


class ResponseCache:
    """Provides intelligent caching with multiple strategies"""
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        
        # Memory-based fallback if diskcache not available
        self._memory_cache: Dict[str, Any] = {}
        self._memory_metadata: Dict[str, CacheEntry] = {}
        
        # Initialize diskcache storage if available
        if dc:
            try:
                self._cache_dir = Path(self.config.cache_dir)
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Main cache for exact matches
                self._main_cache = dc.Cache(str(self._cache_dir / "main"), size_limit=self.config.max_cache_size_mb * 1024 * 1024)
                
                # Similarity cache for semantic matches
                if self.config.enable_similarity_cache:
                    self._similarity_cache = dc.Cache(str(self._cache_dir / "similarity"), size_limit=self.config.max_similarity_cache_size * 1024 * 1024)
                else:
                    self._similarity_cache = None
                
                # Metadata storage
                self._metadata_cache = dc.Cache(str(self._cache_dir / "metadata"), size_limit=10 * 1024 * 1024)
                
            except Exception as e:
                logger.warning(f"Diskcache initialization failed, using memory-only cache: {e}")
                self._main_cache = None
                self._similarity_cache = None
                self._metadata_cache = None
        else:
            self._main_cache = None
            self._similarity_cache = None
            self._metadata_cache = None
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'exact_matches': 0,
            'similarity_matches': 0,
            'partial_matches': 0,
            'cache_size_mb': 0.0,
            'total_requests': 0,
            'avg_response_time_ms': 0.0
        }
        
        # Performance tracking
        self._response_times: List[float] = []
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info("ResponseCache initialized with config: %s", self.config)
    
    async def start(self):
        """Start the cache system"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("ResponseCache started")
    
    async def stop(self):
        """Stop the cache system"""
        self._shutdown_event.set()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close disk caches
        if self._main_cache:
            self._main_cache.close()
        if self._similarity_cache:
            self._similarity_cache.close()
        if self._metadata_cache:
            self._metadata_cache.close()
        
        logger.info("ResponseCache stopped")
    
    async def get(
        self,
        request: Dict[str, Any],
        strategy: CacheStrategy = CacheStrategy.EXACT_MATCH
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response for request
        
        Args:
            request: Request to look up
            strategy: Cache lookup strategy
        
        Returns:
            Optional[Dict[str, Any]]: Cached response or None
        """
        start_time = time.time()
        self._stats['total_requests'] += 1
        
        try:
            cache_key = self._generate_cache_key(request)
            
            # Exact match lookup
            if strategy == CacheStrategy.EXACT_MATCH:
                result = await self._get_exact_match(cache_key, request)
            elif strategy == CacheStrategy.SEMANTIC_SIMILARITY:
                result = await self._get_similarity_match(cache_key, request)
            elif strategy == CacheStrategy.PARTIAL_MATCH:
                result = await self._get_partial_match(cache_key, request)
            elif strategy == CacheStrategy.FUZZY_MATCH:
                result = await self._get_fuzzy_match(cache_key, request)
            else:
                result = None
            
            # Track performance
            response_time = time.time() - start_time
            self._response_times.append(response_time)
            
            # Update statistics
            if result:
                self._stats['hits'] += 1
                if strategy == CacheStrategy.EXACT_MATCH:
                    self._stats['exact_matches'] += 1
                elif strategy == CacheStrategy.SEMANTIC_SIMILARITY:
                    self._stats['similarity_matches'] += 1
                elif strategy == CacheStrategy.PARTIAL_MATCH:
                    self._stats['partial_matches'] += 1
            else:
                self._stats['misses'] += 1
            
            # Update average response time
            if len(self._response_times) > 100:
                self._response_times = self._response_times[-100:]  # Keep last 100
            
            avg_time = sum(self._response_times) / len(self._response_times)
            self._stats['avg_response_time_ms'] = avg_time * 1000
            
            return result
            
        except Exception as e:
            logger.error("Cache get error: %s", e)
            return None
    
    async def set(
        self,
        request: Dict[str, Any],
        response: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Store response in cache
        
        Args:
            request: Request that generated the response
            response: Response to cache
            ttl_seconds: Time to live in seconds
            metadata: Additional metadata
        
        Returns:
            bool: True if stored successfully
        """
        try:
            cache_key = self._generate_cache_key(request)
            ttl = ttl_seconds or self.config.default_ttl_seconds
            
            # Create cache entry
            cache_entry = CacheEntry(
                key=cache_key,
                hash_key=self._generate_hash_key(request),
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=0,
                ttl_expires=time.time() + ttl,
                size_bytes=len(pickle.dumps(response)),
                content_type='response',
                metadata=metadata or {}
            )
            
            # Compress response if enabled
            if self.config.enable_compression:
                # Note: In a real implementation, you'd use proper compression
                pass
            
            # Store in main cache
            cache_data = {
                'response': response,
                'entry': cache_entry,
                'request': request
            }
            
            if self._main_cache:
                self._main_cache[cache_key] = cache_data
                self._metadata_cache[cache_key] = cache_entry
                
                # Store in similarity cache for semantic matching
                if self.config.enable_similarity_cache:
                    await self._store_similarity_entry(request, response, cache_key)
            else:
                # Memory-only storage
                self._memory_cache[cache_key] = cache_data
                self._memory_metadata[cache_key] = cache_entry
                
                if self.config.enable_similarity_cache:
                    await self._store_similarity_entry(request, response, cache_key)
            
            # Update statistics
            self._stats['cache_size_mb'] = self._calculate_cache_size()
            
            logger.debug("Cached response for key: %s", cache_key[:16])
            return True
            
        except Exception as e:
            logger.error("Cache set error: %s", e)
            return False
    
    async def invalidate(self, pattern: str = None, key: str = None) -> int:
        """
        Invalidate cache entries
        
        Args:
            pattern: Glob pattern for keys to invalidate
            key: Specific key to invalidate
        
        Returns:
            int: Number of entries invalidated
        """
        invalidated = 0
        
        try:
            if key:
                # Invalidate specific key
                if self._main_cache and key in self._main_cache:
                    del self._main_cache[key]
                    invalidated += 1
                
                if self._metadata_cache and key in self._metadata_cache:
                    del self._metadata_cache[key]
                
                if self._similarity_cache and key in self._similarity_cache:
                    del self._similarity_cache[key]
                
                # Memory cache
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    invalidated += 1
                
                if key in self._memory_metadata:
                    del self._memory_metadata[key]
            
            elif pattern:
                # Invalidate by pattern (simplified implementation)
                keys_to_remove = []
                
                # Check disk caches
                if self._main_cache:
                    for cached_key in self._main_cache:
                        if pattern in cached_key:
                            keys_to_remove.append(cached_key)
                
                # Check memory cache
                for cached_key in self._memory_cache:
                    if pattern in cached_key:
                        keys_to_remove.append(cached_key)
                
                for key_to_remove in keys_to_remove:
                    if self._main_cache and key_to_remove in self._main_cache:
                        del self._main_cache[key_to_remove]
                        invalidated += 1
                    
                    if self._metadata_cache and key_to_remove in self._metadata_cache:
                        del self._metadata_cache[key_to_remove]
                    
                    if self._similarity_cache and key_to_remove in self._similarity_cache:
                        del self._similarity_cache[key_to_remove]
                    
                    # Memory cache
                    if key_to_remove in self._memory_cache:
                        del self._memory_cache[key_to_remove]
                        invalidated += 1
                    
                    if key_to_remove in self._memory_metadata:
                        del self._memory_metadata[key_to_remove]
            
            logger.info("Invalidated %d cache entries", invalidated)
            return invalidated
            
        except Exception as e:
            logger.error("Cache invalidation error: %s", e)
            return 0
    
    async def _get_exact_match(self, cache_key: str, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get exact match from cache"""
        if self._main_cache and cache_key in self._main_cache:
            cached_data = self._main_cache[cache_key]
            entry = cached_data['entry']
            
            # Check if expired
            if entry.is_expired():
                del self._main_cache[cache_key]
                if self._metadata_cache and cache_key in self._metadata_cache:
                    del self._metadata_cache[cache_key]
                return None
            
            # Update access statistics
            entry.access()
            self._metadata_cache[cache_key] = entry
            
            return cached_data['response']
        
        # Check memory cache
        if cache_key in self._memory_cache:
            cached_data = self._memory_cache[cache_key]
            entry = cached_data['entry']
            
            if entry.is_expired():
                del self._memory_cache[cache_key]
                if cache_key in self._memory_metadata:
                    del self._memory_metadata[cache_key]
                return None
            
            # Update access statistics
            entry.access()
            self._memory_metadata[cache_key] = entry
            
            return cached_data['response']
        
        return None
    
    async def _get_similarity_match(self, cache_key: str, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get similarity match from cache"""
        if not self._similarity_cache and not hasattr(self, '_memory_similarity_cache'):
            return None
        
        # Get semantic features of request
        request_features = self._extract_semantic_features(request)
        
        # Find most similar cached request
        best_match = None
        best_similarity = 0.0
        
        try:
            # Check similarity cache
            if self._similarity_cache:
                for cached_key, cached_features in self._similarity_cache.items():
                    similarity = self._calculate_similarity(request_features, cached_features)
                    
                    if similarity > best_similarity and similarity >= self.config.similarity_threshold:
                        best_similarity = similarity
                        best_match = cached_key
            else:
                # Memory similarity cache
                for cached_key, cached_features in getattr(self, '_memory_similarity_cache', {}).items():
                    similarity = self._calculate_similarity(request_features, cached_features)
                    
                    if similarity > best_similarity and similarity >= self.config.similarity_threshold:
                        best_similarity = similarity
                        best_match = cached_key
                        
        except Exception as e:
            logger.warning("Similarity cache iteration error: %s", e)
            return None
        
        if best_match:
            # Check if the best match is still valid
            if (self._main_cache and best_match in self._main_cache):
                cached_data = self._main_cache[best_match]
                entry = cached_data['entry']
                
                if not entry.is_expired():
                    # Update access statistics
                    entry.access()
                    if self._metadata_cache:
                        self._metadata_cache[best_match] = entry
                    
                    logger.debug("Found similarity match (%.2f%% similar)", best_similarity * 100)
                    return cached_data['response']
            
            # Check memory cache
            elif best_match in self._memory_cache:
                cached_data = self._memory_cache[best_match]
                entry = cached_data['entry']
                
                if not entry.is_expired():
                    # Update access statistics
                    entry.access()
                    self._memory_metadata[best_match] = entry
                    
                    logger.debug("Found similarity match (%.2f%% similar)", best_similarity * 100)
                    return cached_data['response']
        
        return None
    
    async def _get_partial_match(self, cache_key: str, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get partial match from cache"""
        # Look for partial matches in request content
        request_content = self._extract_content_string(request)
        
        try:
            # Check disk cache
            if self._main_cache:
                for cached_key, cached_data in self._main_cache.items():
                    if isinstance(cached_data, dict) and 'entry' in cached_data:
                        entry = cached_data['entry']
                        
                        if entry.is_expired():
                            continue
                        
                        cached_request = cached_data['request']
                        cached_content = self._extract_content_string(cached_request)
                        
                        # Check for partial content overlap
                        if self._calculate_content_overlap(request_content, cached_content) > 0.5:
                            # Update access statistics
                            entry.access()
                            if self._metadata_cache:
                                self._metadata_cache[cached_key] = entry
                            
                            logger.debug("Found partial match")
                            return cached_data['response']
            
            # Check memory cache
            for cached_key, cached_data in self._memory_cache.items():
                if isinstance(cached_data, dict) and 'entry' in cached_data:
                    entry = cached_data['entry']
                    
                    if entry.is_expired():
                        continue
                    
                    cached_request = cached_data['request']
                    cached_content = self._extract_content_string(cached_request)
                    
                    # Check for partial content overlap
                    if self._calculate_content_overlap(request_content, cached_content) > 0.5:
                        # Update access statistics
                        entry.access()
                        self._memory_metadata[cached_key] = entry
                        
                        logger.debug("Found partial match")
                        return cached_data['response']
        except Exception as e:
            logger.warning("Partial match iteration error: %s", e)
        
        return None
    
    async def _get_fuzzy_match(self, cache_key: str, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get fuzzy match from cache"""
        # Simplified fuzzy matching - could be enhanced with proper algorithms
        request_str = self._extract_content_string(request)
        
        try:
            # Check disk cache
            if self._main_cache:
                for cached_key, cached_data in self._main_cache.items():
                    if isinstance(cached_data, dict) and 'entry' in cached_data:
                        entry = cached_data['entry']
                        
                        if entry.is_expired():
                            continue
                        
                        cached_request = cached_data['request']
                        cached_content = self._extract_content_string(cached_request)
                        
                        # Simple fuzzy matching using string similarity
                        similarity = self._calculate_string_similarity(request_str, cached_content)
                        
                        if similarity > 0.7:  # Threshold for fuzzy matching
                            # Update access statistics
                            entry.access()
                            if self._metadata_cache:
                                self._metadata_cache[cached_key] = entry
                            
                            logger.debug("Found fuzzy match (%.2f%% similar)", similarity * 100)
                            return cached_data['response']
            
            # Check memory cache
            for cached_key, cached_data in self._memory_cache.items():
                if isinstance(cached_data, dict) and 'entry' in cached_data:
                    entry = cached_data['entry']
                    
                    if entry.is_expired():
                        continue
                    
                    cached_request = cached_data['request']
                    cached_content = self._extract_content_string(cached_request)
                    
                    # Simple fuzzy matching using string similarity
                    similarity = self._calculate_string_similarity(request_str, cached_content)
                    
                    if similarity > 0.7:  # Threshold for fuzzy matching
                        # Update access statistics
                        entry.access()
                        self._memory_metadata[cached_key] = entry
                        
                        logger.debug("Found fuzzy match (%.2f%% similar)", similarity * 100)
                        return cached_data['response']
        except Exception as e:
            logger.warning("Fuzzy match iteration error: %s", e)
        
        return None
    
    async def _store_similarity_entry(self, request: Dict[str, Any], response: Dict[str, Any], cache_key: str):
        """Store entry in similarity cache"""
        if not self._similarity_cache:
            # Use memory-based similarity cache
            if not hasattr(self, '_memory_similarity_cache'):
                self._memory_similarity_cache = {}
            
            features = self._extract_semantic_features(request)
            self._memory_similarity_cache[cache_key] = features
            return
        
        features = self._extract_semantic_features(request)
        self._similarity_cache[cache_key] = features
    
    def _generate_cache_key(self, request: Dict[str, Any]) -> str:
        """Generate cache key from request"""
        # Create canonical representation
        canonical_request = self._canonicalize_request(request)
        request_str = json.dumps(canonical_request, sort_keys=True)
        
        # Hash the request
        hash_obj = hashlib.sha256(request_str.encode())
        return hash_obj.hexdigest()
    
    def _generate_hash_key(self, request: Dict[str, Any]) -> str:
        """Generate hash key for semantic matching"""
        return hashlib.md5(json.dumps(request, sort_keys=True).encode()).hexdigest()
    
    def _canonicalize_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create canonical representation of request"""
        # Remove non-essential fields that shouldn't affect caching
        canonical = request.copy()
        
        # Remove timestamps and request IDs
        fields_to_remove = ['timestamp', 'request_id', 'session_id', 'user_id']
        for field in fields_to_remove:
            canonical.pop(field, None)
        
        # Normalize whitespace in content
        if 'content' in canonical:
            canonical['content'] = ' '.join(str(canonical['content']).split())
        
        return canonical
    
    def _extract_content_string(self, request: Dict[str, Any]) -> str:
        """Extract content string from request for matching"""
        content_parts = []
        
        # Extract various content fields
        for field in ['content', 'message', 'prompt', 'query']:
            if field in request:
                content_parts.append(str(request[field]))
        
        return ' '.join(content_parts).lower()
    
    def _extract_semantic_features(self, request: Dict[str, Any]) -> Dict[str, float]:
        """Extract semantic features from request"""
        content = self._extract_content_string(request)
        
        # Simple feature extraction (could be enhanced with proper NLP)
        features = {}
        
        # Word count features
        words = content.split()
        features['word_count'] = len(words)
        features['avg_word_length'] = sum(len(word) for word in words) / max(len(words), 1)
        
        # Content type features
        features['has_code'] = 1.0 if '```' in content or '<code>' in content else 0.0
        features['has_question'] = 1.0 if '?' in content else 0.0
        features['has_url'] = 1.0 if 'http' in content else 0.0
        
        # Sentiment indicators (simplified)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disaster']
        
        features['positive_sentiment'] = sum(1 for word in positive_words if word in content) / max(len(words), 1)
        features['negative_sentiment'] = sum(1 for word in negative_words if word in content) / max(len(words), 1)
        
        return features
    
    def _calculate_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate similarity between feature vectors"""
        if not features1 or not features2:
            return 0.0
        
        # Cosine similarity
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        dot_product = sum(features1[key] * features2[key] for key in common_keys)
        norm1 = sum(val ** 2 for val in features1.values()) ** 0.5
        norm2 = sum(val ** 2 for val in features2.values()) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity (simplified)"""
        if not str1 or not str2:
            return 0.0
        
        # Simple similarity based on common words
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _calculate_content_overlap(self, content1: str, content2: str) -> float:
        """Calculate content overlap ratio"""
        if not content1 or not content2:
            return 0.0
        
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        return len(intersection) / max(len(words1), len(words2))
    
    def _calculate_cache_size(self) -> float:
        """Calculate total cache size in MB"""
        total_size = 0.0
        
        # Calculate memory cache size
        total_size += sum(len(pickle.dumps(data)) for data in self._memory_cache.values()) / (1024 * 1024)
        
        # Calculate disk cache size if available
        if self._main_cache:
            total_size += self._main_cache.volume() / (1024 * 1024)
        
        if self._similarity_cache:
            total_size += self._similarity_cache.volume() / (1024 * 1024)
        
        return total_size
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_cleanup()
                await asyncio.sleep(self.config.gc_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup loop: %s", e)
    
    async def _perform_cleanup(self):
        """Perform cache cleanup"""
        try:
            current_time = time.time()
            
            # Remove expired entries from memory cache
            expired_keys = []
            for key, entry in self._memory_metadata.items():
                if entry.is_expired(current_time):
                    expired_keys.append(key)
            
            for key in expired_keys:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                if key in self._memory_metadata:
                    del self._memory_metadata[key]
                if hasattr(self, '_memory_similarity_cache') and key in self._memory_similarity_cache:
                    del self._memory_similarity_cache[key]
            
            self._stats['evictions'] += len(expired_keys)
            
            # Remove expired entries from disk caches
            if self._metadata_cache:
                expired_keys = []
                for key, entry in self._metadata_cache.items():
                    if entry.is_expired(current_time):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    if key in self._main_cache:
                        del self._main_cache[key]
                    if key in self._similarity_cache:
                        del self._similarity_cache[key]
                    del self._metadata_cache[key]
                
                self._stats['evictions'] += len(expired_keys)
            
            # Evict based on policy
            await self._evict_entries()
            
            # Update cache size
            self._stats['cache_size_mb'] = self._calculate_cache_size()
            
            logger.debug("Cleanup completed: %d expired entries removed", len(expired_keys))
            
        except Exception as e:
            logger.error("Cleanup error: %s", e)
    
    async def _evict_entries(self):
        """Evict entries based on policy"""
        # Note: diskcache handles eviction automatically based on size limits
        # Custom eviction policies would be implemented here for advanced cases
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self._stats.copy()
        stats['hit_rate'] = stats['hits'] / max(stats['total_requests'], 1)
        stats['cache_size_mb'] = self._calculate_cache_size()
        return stats
    
    def reset_statistics(self):
        """Reset cache statistics"""
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'exact_matches': 0,
            'similarity_matches': 0,
            'partial_matches': 0,
            'cache_size_mb': 0.0,
            'total_requests': 0,
            'avg_response_time_ms': 0.0
        }
        self._response_times.clear()


# Global cache instance
response_cache = ResponseCache()

# Convenience functions
async def get_response_cache() -> ResponseCache:
    """Get the global response cache instance"""
    await response_cache.start()
    return response_cache


async def cache_response(
    request: Dict[str, Any],
    response: Dict[str, Any],
    strategy: CacheStrategy = CacheStrategy.EXACT_MATCH,
    **kwargs
) -> bool:
    """Cache a response using global cache"""
    cache = await get_response_cache()
    return await cache.set(request, response, **kwargs)


async def get_cached_response(
    request: Dict[str, Any],
    strategy: CacheStrategy = CacheStrategy.EXACT_MATCH
) -> Optional[Dict[str, Any]]:
    """Get cached response using global cache"""
    cache = await get_response_cache()
    return await cache.get(request, strategy)


def create_cache_config(
    cache_dir: str = "/tmp/agent_ui_cache",
    max_size_mb: int = 500,
    default_ttl: int = 3600,
    **kwargs
) -> CacheConfig:
    """Create cache configuration"""
    return CacheConfig(
        cache_dir=cache_dir,
        max_cache_size_mb=max_size_mb,
        default_ttl_seconds=default_ttl,
        **kwargs
    )
