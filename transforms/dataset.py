class AttachEdgeLabel(object):
    def __call__(self, data):
        edge_label, _ = data.y[data.edge_index].min(dim=0)
        data.edge_label = edge_label

        return data
