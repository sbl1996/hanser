class Node:

    def __init__(self, class_name, config, name, inbound_nodes):
        self.class_name = class_name
        self.config = config
        self.name = name
        self.inbound_nodes = inbound_nodes

        for k, v in config.items():
            setattr(self, k, v)

    def __repr__(self):
        return "%s: %s" % (self.class_name, self.name)


class Block:

    def __init__(self):
        pass

    def register(self, name, info):
        splits = name.split('/', 1)
        if len(splits) == 2:
            n, rest = splits
            if hasattr(self, n):
                getattr(self, n).register(rest, info)
            else:
                b = Block()
                b.register(rest, info)
                setattr(self, n, b)
        else: # Node
            n = splits[0]
            if hasattr(self, n):
                raise ValueError("Duplicate %s" % name)
            else:
                setattr(self, n, Node(**info))

    def get_layer(self, name):
        splits = name.split('/', 1)
        if len(splits) == 1:
            return getattr(self, splits[0])
        else:
            return getattr(self, splits[0]).get_layer(splits[1])


class VModel(Block):

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __init__(self, name, layers, input_layers, output_layers):
        super().__init__()
        self.name = name
        self.layers = layers
        self.input_layers = input_layers
        self.output_layers = output_layers

        for l in self.layers:
            self.register(l['name'], l)
            rl = self.get_layer(l['name'])
            inbound_nodes = rl.inbound_nodes
            if inbound_nodes:
                inbound_nodes = inbound_nodes[0]
            rl.inbound_nodes = [
                self.get_layer(n[0])
                for n in inbound_nodes
            ]

    def update(self):
        for l in self.layers:
            node = self.get_layer(l['name'])
            for k in l['config'].items():
                l['config'][k] = getattr(node, k)

    def get_config(self):
        self.update()
        d = {
            'name': self.name,
            'layers': self.layers,
            'input_layers': self.input_layers,
            'output_layers': self.output_layers,
        }
        return d
#
# d = read_json('/Users/hrvvi/Downloads/resnet50.json')
# m = VModel.from_config(d)