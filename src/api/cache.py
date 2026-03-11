"""
Production-grade caching system for YASEN-ALPHA API
Thread-safe, background updates, zero dependencies
"""

import time
import threading
import json
from datetime import datetime
from typing import Any, Optional, Callable

class ProductionCache:
    """
    Enterprise-level cache with background updates
    """
    
    def __init__(self, ttl_seconds: int = 300):
        self._cache = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self.ttl = ttl_seconds
        self._stats = {
            'hits': 0,
            'misses': 0,
            'updates': 0
        }
        self._running = False
        self._updater_thread = None
        
    def get(self, key: str) -> Optional[Any]:
        """Thread-safe get with stats tracking"""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    self._stats['hits'] += 1
                    return value
                else:
                    # Expired - remove it
                    del self._cache[key]
            
            self._stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any):
        """Thread-safe set"""
        with self._lock:
            self._cache[key] = (value, time.time())
    
    def get_stats(self) -> dict:
        """Get cache performance stats"""
        with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total * 100) if total > 0 else 0
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': f"{hit_rate:.1f}%",
                'cache_size': len(self._cache)
            }
    
    def start_background_updates(self, update_func: Callable):
        """Start background thread for cache updates"""
        self._running = True
        self._updater_thread = threading.Thread(
            target=self._updater_worker,
            args=(update_func,),
            daemon=True,
            name="CacheUpdater"
        )
        self._updater_thread.start()
        print(f"✅ Cache updater started (TTL: {self.ttl}s)")
    
    def _updater_worker(self, update_func: Callable):
        """Background worker that keeps cache fresh"""
        while self._running:
            try:
                # Update all cache keys
                fresh_data = update_func()
                if fresh_data:
                    for key, value in fresh_data.items():
                        self.set(key, value)
                    with self._lock:
                        self._stats['updates'] += 1
                    print(f"🔄 Cache updated at {datetime.now().strftime('%H:%M:%S')}")
            except Exception as e:
                print(f"⚠️ Cache update error: {e}")
            
            # Wait for next update
            time.sleep(self.ttl)
    
    def stop(self):
        """Stop background updates"""
        self._running = False
        if self._updater_thread:
            self._updater_thread.join(timeout=2)

# Global cache instance (singleton)
cache = ProductionCache(ttl_seconds=300)