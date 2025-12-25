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
        evict_keys: list[KeyType] = []

        # Safety limit to prevent infinite loops if all items are pinned.
        # In the worst case we may need to "cycle" through evictable entries
        # multiple times until their access counts are decremented to 0.
        max_iterations = len(cache_dict) * (self.max_access_count + 2)
        iterations = 0

        while len(evict_keys) < num_candidates and cache_dict:
            if iterations > max_iterations:
                logger.warning(
                    "Max iterations reached in get_evict_candidates. Stopping search."
                )
                break

            # Find the oldest *evictable* entry (pinned entries are skipped but not
            # reordered; FIFO order among pinned entries is preserved).
            selected_key: KeyType | None = None
            for key in list(cache_dict.keys()):
                if key in evict_keys:
                    continue
                cache_val = cache_dict.get(key)
                if cache_val is None:
                    continue
                if not cache_val.can_evict:
                    continue
                selected_key = key
                break

            if selected_key is None:
                # Best-effort: nothing currently evictable.
                break

            count = self.access_counts.get(selected_key, 0)
            if count > 0:
                # Reinsertion step: decrement the access count and move to tail.
                self.access_counts[selected_key] = count - 1
                cache_dict.move_to_end(selected_key)
            else:
                # Candidate found.
                evict_keys.append(selected_key)
                # IMPORTANT: many backends evict via `batched_remove(..., force=False)`
                # which does not call `update_on_force_evict`. Clean up here to avoid
                # leaking internal state.
                self.access_counts.pop(selected_key, None)

            iterations += 1

        return evict_keys
