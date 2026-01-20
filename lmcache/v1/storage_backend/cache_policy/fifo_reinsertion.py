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

        # We allow multiple rounds (bounded by max_access_count) so items with a
        # positive access counter can age out without unbounded scanning. Each
        # round is a single pass over cache_dict; reinsertions are applied after
        # the pass to keep iteration safe.
        max_rounds = self.max_access_count + 1
        for _ in range(max_rounds):
            keys_to_reinsert: list[KeyType] = []

            for key, cache_val in cache_dict.items():
                if cache_val is None:
                    continue
                if not cache_val.can_evict:
                    continue

                count = self.access_counts.get(key, 0)
                if count > 0:
                    self.access_counts[key] = count - 1
                    keys_to_reinsert.append(key)
                    continue

                evict_keys.append(key)
                # IMPORTANT: many backends evict via `batched_remove(..., force=False)`
                # which does not call `update_on_force_evict`. Clean up here to avoid
                # leaking internal state.
                self.access_counts.pop(key, None)

                if len(evict_keys) == num_candidates:
                    break

            # Apply reinsertion after iteration to avoid mutating during traversal.
            for key in keys_to_reinsert:
                cache_dict.move_to_end(key)

            if len(evict_keys) == num_candidates:
                break
            if not keys_to_reinsert:
                # No progress possible (all pinned or None).
                break

        return evict_keys
