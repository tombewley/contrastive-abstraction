import networkx as nx
import numpy as np

class Graph:
    """Graph defined by transition counts matrix."""
    def __init__(self, C, bb): 
        self.graph = nx.DiGraph()
        # Create nodes.
        self.m = C.shape[0]
        C_marg_max = C.sum(axis=1).max()
        self.graph.add_nodes_from([(i, {"bb": bb[i], "alpha": C[i,:].sum() / C_marg_max} if i < self.m else {"bb": None, "alpha": 1}) for i in range(self.m+1)])
        # Create edges.
        C_ns = C.copy(); r = np.arange(self.m); C_ns[r,r] = 0; max_C_ns = C_ns.max()
        for i in range(self.m): 
            for j in range(self.m+1):
                if i != j and C[i,j] > 0:
                    self.graph.add_edge(i, j, 
                        C = int(C[i,j]), # Get rid of NumPy datatypes to allow JSON serialisation.
                        # P = float(P[i,j]),
                        # T = float(T[i,j]),
                        alpha = float(C[i,j] / max_C_ns),
                        # cost = float(-np.log(T[i,j])) # Edge cost = negative log conditional prob.
                        )

    def dijkstra(self, source, dest=None):
        """
        Use networkx's inbuilt Dijktra algorithm to find the highest-probability paths from a source leaf.
        If a destination is specified, use that. Otherwise, find paths to all other leaves.
        """
        return nx.single_source_dijkstra(self.graph, source=source, target=dest, weight="cost")

    def json(self, *attributes, clip=None):
        """
        Create json serialisable representation of graph.
        """
        g = self.graph.copy()
        relabel = {}; reattr = {}
        for node, attr in g.nodes(data=True):
            # Replace node with its index to make serialisable. 
            relabel[node] = attr["idx"]; del attr["idx"]
            try: 
                # Collect node attributes.
                reattr[node] = {**attr, **node.json(*attributes, clip=clip)}
            except: continue # For initial/terminal.
        nx.set_node_attributes(g, reattr)
        g = nx.relabel_nodes(g, relabel)
        j = nx.readwrite.json_graph.node_link_data(g)
        j["var_names"] = self.space.dim_names
        return j

    def show(self, layout_dims=None, terminal=False, highlight_path=None, alpha=True, ax=None):
        """
        Visualise graph using networkx.
        """
        NODE_SIZE = 30 # 50
        COLOUR = "k"; LINE_WIDTH = 0.369

        import matplotlib.pyplot as plt
        if ax is None: _, ax = plt.subplots()#figsize=(12,12))   
        if terminal: nodelist, edgelist = None, self.graph.edges
        else: nodelist, edgelist = [n for n in self.graph.nodes if n != self.m], [e for e in self.graph.edges if e[1] != self.m]
        if layout_dims is not None:
            pos = {}
            bbs = nx.get_node_attributes(self.graph, "bb")
            for node in self.graph.nodes(): 
                bb = bbs[node]
                if bb is not None: pos[node] = (bb[layout_dims,0] + bb[layout_dims,1])/2
            pos_array = np.vstack(tuple(pos.values()))
            pos_1_mean = np.mean(pos_array[:,1])
            pos[self.m] = [9,9] # [np.max(pos_array[:,0]), pos_1_mean]
        else: 
            # If no layout_dims, arrange using spring forces.               
            #pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog="neato")#, args="-len=5 -Gmaxiter=10000") 
            pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog="sfdp", args="-Grepulsiveforce=7.5") 
        # Draw nodes and labels.
        alpha = nx.get_node_attributes(self.graph, "alpha"); alpha = [alpha[n] for n in range(self.m)]
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=nodelist, ax=ax,
                               node_size=NODE_SIZE,
                               node_color=[COLOUR for _ in range(self.m)] + (["r"] if terminal else []),
                               alpha=alpha,
                               edgecolors=COLOUR, linewidths=LINE_WIDTH)
        nx.draw_networkx_labels(self.graph, labels=(None if terminal else {n:n+1 for n in nodelist}), pos=pos, ax=ax,
                                font_family="ETBembo", 
                                font_size=5.5, 
                                font_color="k",
                                )
        # If highlight_path specified, highlight it in a different colour.
        if highlight_path is not None:
            h = set((highlight_path[i], highlight_path[i+1]) for i in range(len(highlight_path)-1))
            edge_colours = []
            for edge in edgelist:
                if edge in h: edge_colours.append("r")
                else: edge_colours.append("k")
        else: edge_colours = ["k" for _ in edgelist]
        arcs = nx.draw_networkx_edges(self.graph, pos=pos, edgelist=edgelist, ax=ax,
                                      node_size=NODE_SIZE,
                                      width=LINE_WIDTH,
                                      edge_color=edge_colours,
                                      arrowsize=10*LINE_WIDTH,
                                      connectionstyle="arc3,rad=0.2"
                                      )
        # Set alpha individually for each non-highlighted edge.
        if alpha:
            for edge, arc, c in zip(edgelist, arcs, edge_colours):
                if c != "r": arc.set_alpha(self.graph.get_edge_data(*edge)["alpha"])
        # Retrieve axis ticks which networkx likes to hide.
        if layout_dims is not None: 
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        return ax