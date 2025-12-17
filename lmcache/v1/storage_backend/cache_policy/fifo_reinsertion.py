# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy, KeyType

logger = init_logger(__name__)


class FIFOReinsertionCachePolicy(BaseCachePolicy[KeyType, OrderedDict[KeyType, Any]]):
    """
    FIFO-Reinsertion cache policy.
    """

    def __init__(self, max_access_count: int = 3):
        logger.info(f"Initializing FIFOReinsertionCachePolicy with max_access_count={max_access_count}")
        self.max_access_count = max_access_count
        self.access_counts: dict[KeyType, int] = {}

    def init_mutable_mapping(self) -> OrderedDict[KeyType, Any]:
        return OrderedDict()

    def update_on_hit(
        self,
        key: KeyType,
        cache_dict: OrderedDict[KeyType, Any],
    ) -> None:
        # Increment access count, capped at max_access_count
        current_count = self.access_counts.get(key, 0)
        self.access_counts[key] = min(current_count + 1, self.max_access_count)
        # Do NOT move to end on hit (unlike LRU)

    def update_on_put(
        self,
        key: KeyType,
    ) -> None:
        # Initialize access count to 0
        self.access_counts[key] = 0

    def update_on_force_evict(
        self,
        key: KeyType,
    ) -> None:
        self.access_counts.pop(key, None)

    def get_evict_candidates(
        self,
        cache_dict: OrderedDict[KeyType, Any],
        num_candidates: int = 1,
    ) -> list[KeyType]:
        evict_keys = []
        
        # Safety limit to prevent infinite loops if all items are pinned
        # or if we somehow get stuck.
        # In a worst case, we might cycle through the whole list multiple times.
        # But counts decrease, so we should terminate unless everything is pinned.
        max_iterations = len(cache_dict) * (self.max_access_count + 2)
        iterations = 0

        while len(evict_keys) < num_candidates and len(cache_dict) > 0:
            if iterations > max_iterations:
                logger.warning("Max iterations reached in get_evict_candidates. Stopping search.")
                break
            
            # Peek at the head of the queue
            try:
                key = next(iter(cache_dict))
            except StopIteration:
                break

            cache_val = cache_dict[key]

            # If the item is already selected for eviction (in this call), skip it?
            # But we move selected items to the end to proceed.
            # So if we see it again, it means we cycled through everything?
            if key in evict_keys:
                # We wrapped around and found our own candidates.
                # This implies we can't find more candidates.
                break

            if not cache_val.can_evict:
                # Cannot evict. Move to end to check next item.
                # This changes the order of pinned items, but it's necessary to proceed.
                cache_dict.move_to_end(key)
                iterations += 1
                continue

            count = self.access_counts.get(key, 0)
            
            if count > 0:
                # Reinsert: Decrement count and move to end
                self.access_counts[key] = count - 1
                cache_dict.move_to_end(key)
                iterations += 1
            else:
                # Found a candidate (count == 0)
                evict_keys.append(key)
                # Move to end to expose the next item for the next candidate search
                cache_dict.move_to_end(key)
                iterations += 1

        return evict_keys
