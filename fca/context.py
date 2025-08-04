class Context:
    def __init__(self, object: int , attr: int, relation: bool):
        self.object = object
        self.attr = attr
        self.relation = 1 if relation else 0