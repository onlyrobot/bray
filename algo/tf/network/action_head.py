from __future__ import annotations


class ActionHead:
    def __init__(self, name, mask_feature_name=None, source_feature_name=None, dependency=None, auto_regressive=False):
        self.name = name
        self.mask_feature_name = mask_feature_name
        self.source_feature_name = source_feature_name
        self.dependency = dependency
        self.auto_regressive = auto_regressive

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: ActionHead) -> bool:
        return hasattr(other, "name") and self.name == other.name

    def __lt__(self, other: ActionHead) -> bool:
        return self.name < other.name
