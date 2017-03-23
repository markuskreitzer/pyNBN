# coding=utf-8
import numpy as np

"""
function topo=gen_topo(type, network)
topo=[];
nl=length(network);
for i=2:nl                  %   for number of layers
    s=sum(network(1:i-1));      %   starting a new layer
    for j=1:network(i)          %   in each layer
        switch type
            case 'SLP'
                topo=[topo, s+j, s-network(i-1)+1:s];
            case 'MLP'
                topo=[topo, s+j, s-network(i-1)+1:s];
            case 'FCC'
                topo=[topo, s+j, 1:s];   %s+j node number and j is always 1
            case 'BMLP'
                topo=[topo, s+j, 1:s];   %s+j node number
        end;
    end;
end;
return;

"""


class Topography(object):
    """
    MLP network => ninp 3 4 2 1
    SLP network => ninp 17 1
    FCC network => ninp 1 1 1 1 1 1
    MLP network => ninp 3 4 2 1
    """

    def __init__(self, topo_type, network):
        self.topo_type = topo_type
        self.network = network
        self.number_of_layers = self.network.shape[0]

    @property
    def types(self):
        return {
            'SLP': self.mlp,
            'MLP': self.mlp,

            'FCC': self.fcc,

            'BMLP': lambda topo, s, j, i: np.concatenate((topo, s + j, range(1, s)), axis=1)
        }

    def fcc(self):
        num_inputs = self.network[0]
        output_node = [num_inputs + 1]
        topography = np.array([])

        for layer_num in range(1, self.network.shape[0]):
            inputs = np.arange(1, num_inputs + layer_num)
            layer_layout = np.concatenate((output_node, inputs))
            output_node = [layer_layout[-1] + 2]
            topography = np.concatenate((topography, layer_layout))

        return topography

    def mlp(self):

        def build_layer(num_inputs, start_point):
            # print "Building layer with", num_inputs, "inputs from", start_point
            output = np.array([start_point + num_inputs])
            inputs = np.arange(start_point, output)
            return np.concatenate((output, inputs))

        number_of_inputs = self.network[0]
        topography = build_layer(number_of_inputs, 1)

        for layer_num in range(1, self.number_of_layers - 1):
            number_of_inputs = self.network[layer_num]
            topography = np.concatenate((topography, build_layer(number_of_inputs, topography[-1] + 1)))

        return topography

    def generate(self):
        return self.types[self.topo_type]()


if __name__ == '__main__':
    print "Testing"
    my_network = np.array([7, 1, 1, 1, 1, 1, 1])

    tp = Topography('FCC', my_network)
    topo = tp.generate()
    print topo
    print "FCC shape", topo.shape
    print
    tp = Topography('MLP', my_network)
    topo = tp.generate()
    print topo
    print "MLP shape", topo.shape

