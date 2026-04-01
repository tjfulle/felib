class NodeVariable:
    def __init__(self, name: str):
        self.name = name

    def labels(self, ndim: int) -> list[str]:
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        return isinstance(other, NodeVariable) and getattr(other, "name") == self.name


class SpatialVectorVar(NodeVariable):
    def labels(self, ndim: int) -> list[str]:
        return [f"{self.name}.{ax}" for ax in "xyz"[:ndim]]


class ScalarVar(NodeVariable):
    def labels(self, ndim: int) -> list[str]:
        return [self.name]


X = 0
Y = 1
Z = 2
