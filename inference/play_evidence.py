"""Short-lived visual evidence for opponent table plays.

A table play can be visible for only one recognition frame while the remaining
count badge needs several frames to become stable.  This cache preserves legal
single-frame candidates and lets a later confirmed count drop validate the
candidate by exact card count.

The cache is evidence only: it never commits an action by itself.
"""


class OpponentPlayEvidence:
    SOURCE_PRIORITY = {
        "single_frame": 1,
        "stable_visual": 2,
        "count_trigger": 3,
    }

    def __init__(self, max_age_seconds=6.0):
        self.max_age_seconds = max(0.5, float(max_age_seconds))
        self._items = {"left": [], "right": []}

    def clear(self, side=None):
        if side is None:
            self._items = {"left": [], "right": []}
            return
        if side in self._items:
            self._items[side] = []

    def prune(self, now):
        now = float(now)
        for side in self._items:
            self._items[side] = [
                item
                for item in self._items[side]
                if now - item["last_seen"] <= self.max_age_seconds
            ]

    def observe(
        self,
        side,
        cards,
        now,
        source="single_frame",
        expected_drop=None,
    ):
        if side not in self._items or not cards:
            return None

        cards = str(cards)
        now = float(now)
        expected_drop = (
            None if expected_drop is None else int(expected_drop)
        )
        self.prune(now)

        # Merge nearby identical observations instead of growing one record per
        # screenshot.  A stronger source upgrades the retained evidence.
        for item in reversed(self._items[side]):
            if (
                item["cards"] == cards
                and now - item["last_seen"] <= 1.25
            ):
                item["last_seen"] = now
                item["hits"] += 1
                if self.SOURCE_PRIORITY.get(source, 0) > (
                    self.SOURCE_PRIORITY.get(item["source"], 0)
                ):
                    item["source"] = source
                if expected_drop is not None:
                    item["expected_drop"] = expected_drop
                return dict(item)

        item = {
            "cards": cards,
            "first_seen": now,
            "last_seen": now,
            "hits": 1,
            "source": source,
            "expected_drop": expected_drop,
        }
        self._items[side].append(item)
        return dict(item)

    def best(self, side, drop, now, max_age_seconds=None):
        if side not in self._items:
            return None
        drop = int(drop)
        now = float(now)
        self.prune(now)
        max_age = (
            self.max_age_seconds
            if max_age_seconds is None
            else max(0.0, float(max_age_seconds))
        )

        candidates = []
        for item in self._items[side]:
            age = now - item["last_seen"]
            if age > max_age or len(item["cards"]) != drop:
                continue
            expected_drop = item.get("expected_drop")
            if expected_drop is not None and expected_drop != drop:
                continue
            candidates.append(item)

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda item: (
                self.SOURCE_PRIORITY.get(item["source"], 0),
                item["hits"],
                item["last_seen"],
            ),
        )
        out = dict(best)
        out["age_seconds"] = max(0.0, now - best["last_seen"])
        return out
