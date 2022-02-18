import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
import pydot 
from joblib import dump

class Tree:
    def __init__(self, dim_names, D=None, term_data=[], class_data=[], thresholds=None, classes=None, weight_mode="pop", kernel=None):
        self.dim_names = dim_names
        self.root = Node(0, np.array([[-np.inf, np.inf] for _ in range(len(self.dim_names))])) 
        self.leaves = [self.root]
        self.weight_mode = weight_mode
        if D is None: D = np.empty((0, len(self.dim_names)))
        if thresholds is None: thresholds = [[] for _ in self.dim_names]
        self.populate(D, term_data, class_data, thresholds, classes, kernel)

    def __str__(self): return f"Transition tree:\n\tNum samples={self.C.sum()}\n\tDims={self.dim_names}\n\tm={self.m}\n\tn={self.n}"
    @property
    def m(self): return self.C.shape[0]
    @property
    def n(self): return len(self.classes)
    @property
    def C0(self): # Initialisation counts.
        transitions_init = [[[idx for idx in tt if idx == 0 or self.term_data[idx-1]] for tt in t] for t in self.transitions]
        return count(transitions_init, self.class_data, self.classes, self.class_mode, self.kernel).sum(axis=1)
    def _C(self, init=False):
        if init: # Add row and column for initial.
            m = self.m
            C = np.zeros((m+1, m+2, self.n), dtype=(float if self.class_mode==2 else int))
            C[:m,:m] = self.C[:,:m]; C[:m,m+1] = self.C[:,m]; C[m,:m] = self.C0
            return C 
        else: return self.C
    @property
    def C_ns(self): C = self._C(); C_ns = C.copy(); r = np.arange(self.m); C_ns[r,r,:] = 0; return C_ns
    @property
    def P(self): C = self._C(); return C / C.sum(axis=(0,1))[None,None,:]
    @property
    def P_ns(self): C_ns = self.C_ns; return C_ns / C_ns.sum(axis=(0,1))[None,None,:]
    @property
    def T(self, validate=False): 
        C, m, n = self.C, self.m, self.n
        T = np.zeros((m+1, m+1, n))
        init = C.sum(axis=1) - C.sum(axis=0)[:-1]        
        T[-1,:m] = init / init.sum(axis=0) # Initial.
        T[:m,:] = C / C.sum(axis=1)[:,None,:] # Others.
        if validate:
            assert (init >= 0).all() and (init.sum(axis=0) > 0).all() 
            assert np.isclose(init.sum(axis=0), C[:,-1,:].sum(axis=0)).all()
            assert np.isclose(np.nansum(T, axis=1), 1).all()
        return T
    @property 
    def T_ns(self): C_ns = self.C_ns; return C_ns / C_ns.sum(axis=1)[:,None,:]
    @property
    def N(self): C = self._C(); return C / C[:,-1].sum(axis=0) # Expected number per episode.
    @property
    def posterior(self): 
        with np.errstate(divide="ignore", invalid="ignore"): return self.C / self.C.sum(axis=2)[:,:,None]
    @property
    def bb_global(self): return np.array([self.D.min(axis=0),self.D.max(axis=0)]).T
    @property
    def bb(self): 
        bb_global = self.bb_global
        print(self.leaves[0].bb.shape, bb_global.shape)
        return [np.minimum(np.maximum(leaf.bb, bb_global[0,:]), bb_global[1,:]) for leaf in self.leaves]

    def dict_or_hr_initialise(self, input=None, redim=None): 
        """Initialise self from a dictionary object or hyperrectangles tree."""
        assert self.m == 1
        if type(input) == dict: 
            is_dict = True
            if redim is not None: # "redim" allows renumbering of split dims.
                for v in input.values(): v["split_dim"] = redim[v["split_dim"]]
        else: is_dict = False; assert len(input.space) == len(self.dim_names)
        def _recurse(leaf_num, n): 
            if (is_dict and n in input) or (not(is_dict) and n.split_dim is not None):
                if is_dict: dim, threshold = input[n]["split_dim"], input[n]["split_threshold"]
                else: dim, threshold = n.split_dim, n.split_threshold
                right_leaf_num = self.m
                left, right = self.leaves.pop(leaf_num).split(dim, threshold, right_leaf_num)
                self.leaves.insert(leaf_num, left); self.leaves.append(right)
                self.leaf2idx, self.idx2leaf, self.transitions, self.C = self.copy_and_expand()
                self.pseudosplit(leaf_num, dim, threshold, self.leaf2idx, self.idx2leaf, self.transitions, self.C)
                _recurse(leaf_num, input[n]["left"] if is_dict else n.left)
                _recurse(right_leaf_num, input[n]["right"] if is_dict else n.right)
        _recurse(0, 1 if is_dict else input.root) # Root node must have key of 1 in dict.

    def populate(self, D:np.ndarray, term_data, class_data:list, thresholds, classes=None, kernel=None):
        """Populate tree with a complete dataset, wiping any existing data.""" 
        # First store thresholds. # NOTE: Must be sorted right-to-left.
        assert D.shape[1] == len(thresholds) == len(self.dim_names)
        self.thresholds = [sorted(t, reverse=True) for t in thresholds] 
        # Create empty data containers.
        try: m = self.m
        except: m = 1 # If initialising for the first time.
        self.idx2leaf = []
        self.leaf2idx = [set() for _ in self.leaves]
        self.transitions = [[[] for _ in range(m+1)] for _ in range(m)]
        self.C = np.zeros((m, m+1, 0), dtype=int) 
        # Assemble class information.
        if classes is None: classes = list(np.unique(class_data)) # Implicit classes.
        if all(type(c) == list for c in classes): # Range classes.
            assert all(len(c) == 2 for c in classes), "Range classes must be in [min, max] form."
            self.classes, self.class_mode, self.kernel = classes, 1, None
        elif all(type(c) == str for c in classes): # Partial-membership classes.
            assert len(classes) == 2, "Can only handle two partial-membership classes."
            if len(class_data) > 0: assert min(class_data) >= 0 and max(class_data) <= 1, "Class membership must be in [0,1]."
            self.classes, self.class_mode, self.kernel = classes, 2, None        
            class_data = [np.array([1-mu, mu]) for mu in class_data] # Precompute membership vector.
        else: # Basic classes.
            self.classes, self.class_mode, self.kernel = {k:i for i,k in enumerate(classes)}, 0, None
        # Store data.
        self.D, self.term_data, self.class_data = D, term_data, list(class_data)
        # Call _pop function.
        self._pop(D, term_data, class_data)

    def populate_incremental(self, D:np.ndarray, term_data:list=None, class_data:list=None):
        """Add data to the tree incrementally, without wiping existing data."""
        assert self.class_mode == 0, "Not checked for anything else."
        # Validation.
        N = len(D)
        if class_data is None: class_data = int(max(self.classes)+1) # Append a new class.
        if type(class_data) in {int, str}: class_data = [class_data for _ in range(N)] # Single class specified as integer.
        assert len(class_data) == N
        classes = list(np.unique(class_data))  
        if term_data is None: term_data = [False for _ in range(N-1)] + [True] # Assume only the final sample is terminal (i.e. single episode). 
        assert term_data[-1] # Final sample must be terminal.
        assert len(classes) <= sum(term_data) # Can't have more classes than episodes. 
        # Add new classes.
        if self.class_mode == 0: 
            i = self.n
            for k in classes: 
                if k not in self.classes: self.classes[k] = i; i += 1
        # Append data.
        self.D = np.vstack((self.D, D))
        self.term_data = np.append(self.term_data, term_data)
        self.class_data += list(class_data)    
        # Call _pop function.
        self._pop(D, term_data, class_data)

    def _pop(self, D:np.ndarray, term_data, class_data, validate=False): 
        """Update self.idx2leaf, self.leaf2idx, self.transitions, self.counts and self.weights with new data."""
        assert D.shape[0] == len(term_data) == len(class_data)
        new_N = len(D); prev_N = len(self.D) - new_N; assert prev_N >= 0
        # Set up bidirectional map between leaves <-> sample indices and use to count transitions.
        if self.m == 1: self.idx2leaf += [0 for _ in range(new_N)] # Quick initialisation if tree is empty.
        else: self.idx2leaf += [self.propagate(x).num for x in D]
        new_transitions = [[[] for _ in range(self.m+1)] for _ in range(self.m)]
        for idx, leaf, term in zip(range(prev_N, prev_N+new_N), self.idx2leaf[-new_N:], term_data): 
            self.leaf2idx[leaf].add(idx)
            next_leaf = self.m if term else self.idx2leaf[idx+1]
            new_transitions[leaf][next_leaf].append(idx) 
            self.transitions[leaf][next_leaf].append(idx)
        # Count new transitions only for efficiency.
        new_C = count(new_transitions, self.class_data, self.classes, self.class_mode, self.kernel)
        # Add to counts.
        num_to_add = self.n - self.C.shape[2] # Add columns to self.C if required.
        if num_to_add: self.C = np.append(self.C, np.zeros((self.m, self.m+1, num_to_add), dtype=(float if self.class_mode in {2,3} else int)), axis=2)
        self.C += new_C
        if self.weight_mode == "pop": self.weights = self.C.sum(axis=(0,1)) / self.C.sum()   
        elif self.weight_mode == "eq": self.weights = np.ones(self.n) / self.n
        if validate:
            assert self.C.shape[2] == self.n
            assert sum([len(tt) for t in self.transitions for tt in t]) == len(self.D)
            if self.weight_mode == "pop": assert np.isclose(self.C.sum(), len(self.D))

    def propagate(self, x):
        def _recurse(node):
            if node.split_dim is None: return node 
            return _recurse(node.right) if x[node.split_dim] >= node.split_threshold else _recurse(node.left)
        return _recurse(self.root)

    def predict(self, trajectory, posterior=False, normalise=True, T=None):
        """Given a trajectory of inputs, map to abstract states then predict class likelihoods or posteriors."""
        states = [self.propagate(t).num for t in trajectory]
        if T is None: T = np.nan_to_num(self.T) # Replaces all NaNs with zeros.
        probs = np.zeros((len(states)+1, self.n))
        if posterior: probs[0] = self.weights # Prior probability.
        else: probs[0] = np.ones(self.n) / self.n
        for t, x in enumerate(states):
            if t == 0: l = self.P[x].sum(axis=0) # Marginal probability for first state.
            else: 



                assert not np.isnan(T[x_prev, x]).any()



                l = T[x_prev, x] # Conditional probability.
            probs[t+1] = probs[t] * l
            x_prev = x
        if normalise: probs = probs / probs.sum(axis=1).reshape(-1,1)
        return probs

    def rollout(self, num_timesteps):
        P_t, P = rollout(self.T, num_timesteps)
        print(f"Rollout error = {np.abs(P - self.P).max()}") # Total variation distance.
        return P_t, P

# ============================================================================
# STAGE 1: OFFLINE STATE ABSTRACTION

    def state_abstract(self, m_max, alpha, post_merge=False, random=False, verbose_level=0, plot_dims=None, colour_classes=False, max_eps_to_plot=np.inf):
        """Run contrastive/random state abstraction algorithm."""            
        if plot_dims is not None: 
            assert len(plot_dims) == 2 # Plotting works for two plot_dims only.
            CLASS_COLOURS = "rgbcymk"
            SPLIT_COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            m_bef = self.m
            plt.figure(figsize=(8,4.8))
            traj_ax = plt.subplot2grid((3, 5), (0, 1), colspan=2, rowspan=2); traj_ax.axis("off")
            jsd_ax_x = plt.subplot2grid((3, 5), (2, 1), colspan=2, sharex = traj_ax)
            jsd_ax_x.ticklabel_format(style="plain", useOffset=False)
            jsd_ax_x.set_xlabel(f"{self.dim_names[plot_dims[0]]} split location"); jsd_ax_x.set_ylabel("JSD")
            jsd_ax_y = plt.subplot2grid((3, 5), (0, 0), rowspan=2, sharey = traj_ax)
            jsd_ax_y.ticklabel_format(style="plain", useOffset=False)
            jsd_ax_y.set_ylabel(f"{self.dim_names[plot_dims[1]]} split location"); jsd_ax_y.set_xlabel("JSD")
            jsd_ax = plt.subplot2grid((3, 5), (0, 3), colspan=2, rowspan=2)
            # jsd_ax.set_xscale("log")
            jsd_ax.set_xlabel("Number of abstract states")
            if max_eps_to_plot > 0:
                for ep, _, p in zip(*self.get_eps(max_eps_to_plot)): 
                    if colour_classes and self.n <= len(CLASS_COLOURS): c = CLASS_COLOURS[p] # If number of classes is small, colour individually.                
                    else: c = [0.8,0.8,0.8]
                    traj_ax.plot(ep[:,plot_dims[0]], ep[:,plot_dims[1]], c=c, lw=0.5)
        # Splitting process for increasing m.
        history_split = [(self.m, None, None, None, jsd(self.C.reshape(-1,self.n).T, self.weights))]
        bb_global = self.bb_global
        while self.m < m_max:
            priority = self.C.sum(axis=(1,2)) # * entropy(tree.C.sum(axis=2), axis=1) # NOTE: Adding entropy weighting here tends to reduce performance!
            print("Exhaustive search over all?")
            ok = False
            for node in priority.argsort()[::-1]:
                gains, max_gain, dim, threshold = self.split(node, 
                                                             min_to_split=history_split[-1][4] + alpha,
                                                            #  min_to_split=history_split[-1] + (1e-12 if post_merge else alpha), 
                                                             random=random,
                                                             verbose=(verbose_level > 1))
                if dim is not None: 
                    history_split.append((self.m, node, dim, threshold, max_gain))
                    if verbose_level > 0: 
                        print(f"{history_split[-1][0]}:\tSplit {history_split[-1][1]} at {self.dim_names[history_split[-1][2]]}={history_split[-1][3]}, JSD = {history_split[-1][4]}")
                    ok = True; break
                elif verbose_level > 1: print(f"\t(No split {node})")
            if not ok: 
                if verbose_level > 1: print("ALPHA OR RESOLUTION THRESHOLD REACHED")
                break
            if plot_dims is not None: 
                c = SPLIT_COLOURS[(self.m-2) % (len(SPLIT_COLOURS)-1)]
                jsd_ax_x.plot([g[1] for g in gains if g[0] == plot_dims[0]], [g[2] for g in gains if g[0] == plot_dims[0]], c=c)
                jsd_ax_y.plot([g[2] for g in gains if g[0] == plot_dims[1]], [g[1] for g in gains if g[0] == plot_dims[1]], c=c)
                leaf_bb = self.leaves[node].bb[plot_dims]
                leaf_bb[:,0] = np.maximum(leaf_bb[:,0], bb_global[plot_dims,0])
                leaf_bb[:,1] = np.minimum(leaf_bb[:,1], bb_global[plot_dims,1])
                if dim == plot_dims[0]:   traj_ax.plot([threshold, threshold], leaf_bb[1], c=c, alpha=1)
                elif dim == plot_dims[1]: traj_ax.plot(leaf_bb[0], [threshold, threshold], c=c, alpha=1)
        # Merging process for decreasing m.
        history_merge = self.recursive_merge(alpha, verbose_level) if post_merge else []
        if plot_dims is not None: 
            self.show(dims=plot_dims, bbs=True, nums=True, ax=traj_ax, lw=2, zorder=10)
            jsd_split = [x[4] for x in history_split]
            m_range = np.arange(m_bef, m_bef+len(jsd_split))
            jsd_ax.plot(m_range, jsd_split, label="JSD")
            jsd_ax.plot(m_range, alpha * (m_range - 1), label="$\\alpha(m-1)$")
            jsd_ax.plot(m_range, np.array(jsd_split) - alpha * (m_range - 1), c="k", label="JSD-$\\alpha(m-1)$")
            if len(history_merge) > 0:
                jsd_merge = reversed([x[4] for x in history_merge])
                jsd_ax.plot(np.arange(self.m, self.m+len(jsd_merge)), jsd_merge)
            jsd_ax.legend()
        return history_split, history_merge

    def get_eps(self, n="all"):
        """Sample a batch of complete episodes and return alongside their class numbers."""
        ep_ends = np.where(self.term_data)[0]
        eps = np.split(self.D ,ep_ends+1)[:-1]
        leaves = np.split(self.idx2leaf, ep_ends+1)[:-1]
        if n == "all": indices = np.arange(len(eps))
        elif type(n) in (list, tuple): indices = n
        else: assert type(n)==int; indices = np.random.choice(len(eps), n, replace=False)
        classes = [self.class_data[0 if i == 0 else (ep_ends[i-1] + 1)] for i in indices]
        if self.class_mode == 0: class_numbers = [self.classes[k] for k in classes]
        elif self.class_mode == 1:
            class_numbers = [] 
            for k in classes: 
                for p, (mn, mx) in enumerate(self.classes):
                    if k >= mn and k < mx: class_numbers.append(p); break      
        elif self.class_mode == 2: class_numbers = classes
        else: raise NotImplementedError()
        return [eps[i] for i in indices], [list(leaves[i]) for i in indices], class_numbers

    def split(self, node, min_to_split, random=False, verbose=False):
        """Find and make best/random split."""
        gains = self.gains(node, None, verbose)
        if random: 
            if len(gains) == 0: return gains, None, None, None
            dim, threshold, max_gain = gains[np.random.randint(len(gains))]
        else:
            max_gain = max([g[2] for g in gains]) if len(gains) > 0 else 0
            if max_gain < min_to_split: return gains, max_gain, None, None
            else: dim, threshold, _ = gains[np.argmax([g[2] for g in gains])]
        left, right = self.leaves.pop(node).split(dim, threshold, self.m)
        self.leaves.insert(node, left); self.leaves.append(right)
        self.leaf2idx, self.idx2leaf, self.transitions, self.C = self.copy_and_expand()
        self.pseudosplit(node, dim, threshold, self.leaf2idx, self.idx2leaf, self.transitions, self.C)
        return gains, max_gain, dim, threshold

    def gains(self, node, alpha, verbose):
        """Evaluate gain from splitting a given node at each valid split threshold."""
        gains = []
        m, n = self.m, self.n
        for dim in range(self.D.shape[1]):
            leaf2idx, idx2leaf, transitions, C = self.copy_and_expand()
            for threshold in self.thresholds[dim]: # Remember that these are sorted right-to-left.
                if threshold >= self.leaves[node].bb[dim,1]: continue # Only consider thresholds within the bb.
                elif threshold <= self.leaves[node].bb[dim,0]: break
                # Perform a pseudosplit.
                self.pseudosplit(node, dim, threshold, leaf2idx, idx2leaf, transitions, C)
                # Compute gain.
                if False: # Jensen-Shannon divergence between left and right transition probabilities
                    counts_here_nt = counts[[node,m],:-1,:]
                    if not np.all(counts_here_nt.sum(axis=(1,2))): continue
                    counts_here_nt[1,[node, m]] = counts_here_nt[1,[m, node]] # Swap rows to enable direct comparison. 
                    w = counts_here_nt.sum(axis=1) # ; nz = w > 0
                    H = entropy(counts_here_nt, axis=1)
                    # print(counts_here_nt)
                    # print(entropy(counts_here_nt.sum(axis=0)) * w.sum(axis=0))
                    # print((H * w).sum(axis=0))
                    j = (entropy(counts_here_nt.sum(axis=0)) * w.sum(axis=0)) - (H * w).sum(axis=0)
                    # print(jsd, np.nansum(jsd))
                    gains.append([dim, threshold, np.nansum(j)])
                elif False: # Weighted entropy sum, ignoring self-loops and terminal.
                    counts_here_nt = counts[[node,m],:-1,:]
                    counts_here_nt[0,node] = 0 # Ignore self-loops.
                    counts_here_nt[1,m] = 0
                    # counts_here_nt[0,m] = 0 # Ignore newly-created transitions.
                    # counts_here_nt[1,node] = 0
                    M = np.hstack(counts_here_nt)
                    w = M.sum(axis=0); nz = w > 0
                    H = entropy(M[:,nz], axis=0)
                    # print(counts_here_nt)
                    print(M)
                    print(w * entropy(M, axis=0))
                    print("H sum here =", np.dot(w[nz], H))
                    gains.append([dim, threshold, -np.dot(w[nz], H)])
                elif False: # Weighted entropy sum, ignoring self-loops and terminal.
                    counts_nt = counts[:,:-1,:]
                    for j in range(n): np.fill_diagonal(counts_nt[:,:,j], 0) # Ignore self-loops.
                    # counts_nt[node,m] = 0 # Ignore newly-created transitions.
                    # counts_nt[m,node] = 0
                    M = np.hstack(counts_nt)
                    w = M.sum(axis=0); nz = w > 0
                    H = entropy(M[:,nz], axis=0)
                    # print(M, M.shape)
                    # print(w[nz] * H)
                    # print("H sum =", np.dot(w[nz], H))
                    gains.append([dim, threshold, -np.dot(w[nz], H)])
                elif True: # Jensen-Shannon divergence between per-class counts.
                    jsd_after = jsd(C.reshape(-1,n).T, self.weights)
                    # if jsd_after - jsd_before >= alpha: 
                    gains.append([dim, threshold, jsd_after])
                if verbose:
                    print(f"{node} @ {dim} = {threshold}")
                    print(np.moveaxis(self.C, 2, 0))
                    print("-----")
                    print(np.moveaxis(C, 2, 0))
                    print("Gain =", gains[-1][2])
                    print("\n")        
        return gains

    def pseudosplit(self, left, dim, threshold, leaf2idx, idx2leaf, transitions, C, validate=False):
        """Simulate splitting a leaf in two, modifying leaf2idx, idx2leaf, transitions and counts C in-place."""
        # Update bidirectional leaf <-> index map.
        m = len(leaf2idx); right = m-1; n = len(self.classes)
        for idx in leaf2idx[left]: 
            if self.D[idx,dim] >= threshold: leaf2idx[right].add(idx); idx2leaf[idx] = right # Add to right.
        leaf2idx[left] -= leaf2idx[right] # Remove from left.
        # Update transitions. 
        for i in range(m): 
            for j in range(m+1): 
                if i != left and j != left: continue # All other counts remain unchanged.
                move, a, b = set(), 0, 0 # a,b bound range of active classes for class_mode 1.
                for idx in transitions[i][j]:
                    i_new, j_new = idx2leaf[idx], m if self.term_data[idx] else idx2leaf[idx+1]
                    if (i != i_new) or (j != j_new):
                        move.add(idx)
                        transitions[i_new][j_new].append(idx)
                        # Update active classes and transfer transition.
                        k = self.class_data[idx]
                        if self.class_mode == 1: # Range mode. NOTE: algorithm assumes both indices and classes are sorted.
                            for kk in range(b,n): # Activate.
                                if k < self.classes[kk][0]: break # Window is a right-open interval.
                                b += 1                
                            for kk in range(a,b): # Deactivate.
                                if k < self.classes[kk][1]: break
                                a += 1
                            C[i,j,a:b] -= 1; C[i_new,j_new,a:b] += 1
                        elif self.class_mode == 2: # Partial membership mode.
                            C[i,j] -= k; C[i_new,j_new] += k
                        elif self.class_mode == 3: # Kernel mode.
                            gaussian = np.exp(-((k-self.classes) / self.kernel)**2).flatten() # Gaussian kernel.
                            C[i,j] -= gaussian; C[i_new,j_new] += gaussian
                        else: # Basic mode.
                            C[i,j,self.classes[k]] -= 1; C[i_new,j_new,self.classes[k]] += 1
                transitions[i][j] = [idx for idx in transitions[i][j] if idx not in move]
        # Post-sort the transitions lists with new items.
        # This is faster than using bisect.insort(), which keeps the lists sorted *at all times*.
        # We only need them to be sorted at the end. 
        for t in transitions[right]: t.sort()
        for t in transitions: t[right].sort()
        # When counts are continuous there's a chance that they might drop slightly below zero, so clip.
        if self.class_mode in {2,3}:
            mn = C.min()
            if mn < 0: 
                assert mn >= -1e-8
                np.clip(C, 0, None, out=C)
        if validate: 
            assert (C >= 0).all()
            assert np.isclose(C.sum(axis=(0,1)), self.C.sum(axis=(0,1))).all()
            np.clip(C, 0, None, out=C)

    def copy_and_expand(self):
        """Copy and expand leaf2idx, idx2leaf, transitions and counts C. Prevents overwriting the stored objects."""
        m, n = self.m, self.n
        leaf2idx = [{idx for idx in i} for i in self.leaf2idx] 
        idx2leaf = [i for i in self.idx2leaf] 
        transitions = [[[t for t in j] for j in i] for i in self.transitions]
        C = np.zeros((m+1,m+2,n), dtype=(float if self.class_mode in {2,3} else int))
        C[:-1,:-2] = self.C[:,:-1]; C[:-1,-1] = self.C[:,-1] 
        # Expand: new leaf (right) appears as the last row and penultimate column.
        leaf2idx.append(set()); transitions.append([[] for _ in range(m+1)])
        for t in transitions: t.insert(-1, []) 
        return leaf2idx, idx2leaf, transitions, C

    def recursive_merge(self, alpha, verbose_level=0):
        """Recursively measure the JSD gain delivered by each leaf pair and prune the least beneficial until alpha threshold met."""
        history = [(self.m, (), jsd(self.C.reshape(-1,self.n).T, self.weights))]
        if verbose_level > 0: print(f"{history[0][0]}:\tJSD = {history[0][2]}")
        while self.m > 1:
            node, j, leaf2idx, idx2leaf, transitions, C = self.pseudomerge_best()
            if history[-1][2] - j > alpha: # Condition for stopping the merging process.
                if verbose_level > 1: print("ALPHA THRESHOLD REACHED")
                break
            history.append((self.m-1, (node.left.num, node.right.num), j))
            if verbose_level > 0: print(f"{history[-1][0]}:\tMerge {history[-1][1][0]} and {history[-1][1][1]}, JSD = {history[-1][2]}")
            self.prune_subtree(node)
            self.leaf2idx, self.idx2leaf, self.transitions, self.C = leaf2idx, idx2leaf, transitions, C                 
        return history

    def pseudomerge_best(self):
        """Find best leaf pair to merge."""
        jsds = []
        for node in self.mergeable():
            leaf2idx, idx2leaf, transitions = self.pseudomerge(node.left.num, node.right.num)
            C = count(transitions, self.class_data, self.classes, self.class_mode, self.kernel)
            jsds.append((node, jsd(C.reshape(-1,self.n).T, self.weights), leaf2idx, idx2leaf, transitions, C))
        jsds.sort(key=lambda x:x[1])
        return jsds[-1]

    def mergeable(self): 
        """Evaluate which pairs of leaves can be merged."""
        if True:
            def _recurse(node, mergeable):
                if node.split_dim is not None: 
                    if node.left.num is not None and node.right.num is not None: mergeable.add(node)
                    else: mergeable = _recurse(node.left, mergeable); mergeable = _recurse(node.right, mergeable)
                return mergeable
            return _recurse(self.root, set())
        else:
            raise Exception("Current merge_pairs implementation doesn't work with tree structure!")
            merge_pairs = []
            num_dims = self.D.shape[1]
            for i in range(self.m):
                for j in range(i):
                    same, adj = [], []
                    for dim in range(num_dims):
                        if (self.leaves[i].bb[dim] == self.leaves[j].bb[dim]).sum() == 2: same.append(dim)
                        elif self.leaves[i].bb[dim,0] == self.leaves[j].bb[dim,1]: adj.append(dim)
                        elif self.leaves[i].bb[dim,1] == self.leaves[j].bb[dim,0]: adj.append(dim)
                    if len(same) == num_dims-1 and len(adj) == 1: merge_pairs.append((j,i))
            return merge_pairs

    def pseudomerge(self, left, right):
        """Simulate merging two leaves, returning modified copies of leaf2idx, idx2leaf and transitions."""        
        assert right > left
        # Copy in order to prevent overwriting the stored objects.
        idx2leaf = [(i if i < right else (left if i == right else i-1)) for i in self.idx2leaf] 
        leaf2idx, transitions = [], []
        for i in range(self.m):
            if i == left:
                leaf2idx.append({idx for idx in self.leaf2idx[left] | self.leaf2idx[right]})
                # Need to re-sort to maintain order when merging.
                transitions.append([sorted(self.transitions[left][left] + self.transitions[left][right] + self.transitions[right][right] + self.transitions[right][left]) if j == left 
                                    else sorted(self.transitions[left][j] + self.transitions[right][j]) for j in range(self.m+1) if j != right])
            elif i != right: 
                leaf2idx.append({idx for idx in self.leaf2idx[i]})
                transitions.append([sorted(self.transitions[i][left] + self.transitions[i][right]) if j == left 
                                    else self.transitions[i][j] for j in range(self.m+1) if j != right])
        return leaf2idx, idx2leaf, transitions

    def prune_subtree(self, node):
        li, ri = node.left.num, node.right.num
        self.leaves[li] = node
        self.leaves.pop(ri)
        node.num, node.left, node.right, node.split_dim, node.split_threshold = li, None, None, None, None
        for l in self.leaves[ri:]: l.num -= 1 # Reindex.

# ============================================================================
# STAGE 2: TEMPORAL ABSTRACTION

    def time_abstract(self, windows=None, n_max=None, beta=None, epsilon=1, random=False, post_merge=False, verbose_level=0, plot=False, validate=False): 
        """Run window-based temporal abstraction algorithm given a fixed state abstraction."""
        assert self.class_mode == 0, "Temporal abstraction must start with class_mode = 0."
        assert self.weight_mode == "pop"
        if windows is None:
            assert n_max is not None and beta is not None
            # Splitting process for increasing n.
            if plot: 
                SPLIT_COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                _, (split_ax, jsd_ax) = plt.subplots(1, 2, figsize=(6.7,3.1), sharey=True)
                # jsd_ax.set_xscale("log")
                split_ax.set_xlabel("Episode")
                split_ax.set_ylabel("JSD")
                jsd_ax.set_xlabel("Number of windows")
            C = self.C.reshape(-1,self.n).T # Flattened.
            windows = {(0, self.n): [C[0:self.n].sum(axis=0), True]}
            jsd_split = [0]
            while len(windows) < n_max:
                # Try splitting every existing window.
                jsd_split_cand = []
                for window in sorted(windows):
                    if not windows[window][1]: continue
                    j = self.window_split(C, windows, window, epsilon, jsd_split[-1])
                    if j == []: windows[window][1] = False # Don't try to split again.
                    else: jsd_split_cand += j
                if len(jsd_split_cand) == 0: 
                    if verbose_level > 0: print("NO MORE SPLITS TO MAKE")
                    break
                if random: 
                    window, threshold, j = jsd_split_cand[np.random.randint(len(jsd_split_cand))]
                else: 
                    window, threshold, j = sorted(jsd_split_cand, key=lambda x:x[2], reverse=True)[0]
                if not post_merge and j - jsd_split[-1] < beta: # Condition for stopping the splitting process.
                    if verbose_level > 0: print("BETA THRESHOLD REACHED")
                    break
                # Split whichever gives highest JSD.
                jsd_split.append(j)
                del windows[window]
                start, end = window
                windows[(start, threshold)] = [C[start:threshold].sum(axis=0), True]
                windows[(threshold, end)] = [C[threshold:end].sum(axis=0), True]
                if verbose_level > 0: print(f"{len(windows)}:\tSplit {window} at {threshold}, JSD = {jsd_split[-1]}")
                if plot: 
                    c = SPLIT_COLOURS[(len(windows)-2) % (len(SPLIT_COLOURS)-1)]
                    split_ax.plot([t for _,t,_ in jsd_split_cand], [j for _,_,j in jsd_split_cand], color=c)
                    split_ax.plot([threshold, threshold], [0, jsd_split[-1]], color=c)
            # Merging process for decreasing n.
            jsd_merge = [jsd_split[-1]]
            if post_merge:
                while len(windows) > 2:
                    # Try merging every consecutive pair of windows.
                    jsd_merge_cand = []
                    window_list = list(sorted(windows))
                    for w1, w2 in zip(window_list[:-1], window_list[1:]):
                        jsd_merge_cand.append((w1, w2, self.window_merge(windows, w1, w2)))
                    # Merge whichever gives highest JSD.
                    w1, w2, j = sorted(jsd_merge_cand, key=lambda x:x[2], reverse=True)[0]
                    if jsd_merge[0] - j > beta: # Condition for stopping the merging process.
                        if verbose_level > 0: print("BETA THRESHOLD REACHED")
                        break 
                    jsd_merge.insert(0, j)
                    windows[(w1[0], w2[1])] = [windows[w1][0] + windows[w2][0], True]
                    del windows[w1]; del windows[w2]
                    if verbose_level > 0: print(f"{len(windows)}:\tMerge {w1} and {w1}, JSD = {jsd_merge[-1]}")
            if plot: 
                n_range = np.arange(1, len(jsd_split)+1)
                jsd_ax.plot(n_range, jsd_split, label="JSD")
                jsd_ax.plot(n_range, beta * (n_range - 1), label="$\\beta(n-1)$")
                jsd_ax.plot(n_range, np.array(jsd_split) - beta * (n_range - 1), c="k", label="JSD-$\\beta(n-1)$")
                # jsd_ax.plot(np.arange(len(windows), len(jsd_split)+1), jsd_merge)
                jsd_ax.legend()
        # Apply windows.
        self.class_mode, self.classes = 1, sorted(windows, key=lambda x: x[0])
        self.C = np.stack([self.C[:,:,w1:w2].sum(axis=2) for w1, w2 in self.classes], axis=2)
        if self.weight_mode == "pop": self.weights = self.C.sum(axis=(0,1)) / self.C.sum()     
        else: raise Exception()
        if validate: 
            assert (self.C == count(self.transitions, self.class_data, self.classes, self.class_mode, self.kernel)).all()
        return jsd_split, jsd_merge, self.classes

    def window_split(self, C, windows, w, epsilon, jsd_last):
        # if len(windows) > 1: C_other = np.vstack([C for ww, (C, _) in windows.items() if ww != w])
        C_split = np.zeros((2, C.shape[1]), dtype=int)
        C_split[1] = windows[w][0]
        assert np.isclose(self.weights.sum(), 1)
        weight = self.weights[w[0]:w[1]].sum() # Proportion of samples in this window.
        jsd_split_cand = []
        for threshold in range(w[0]+1, w[1]):
            C_split[0] += C[threshold-1]; C_split[1] -= C[threshold-1]
            if threshold-w[0] >= epsilon and w[1]-threshold >= epsilon:
                # NOTE: This gives the same ordering over candidates as computing for all windows.
                jsd_split_cand.append((w, threshold, jsd_last+weight*jsd(C_split))) 
        return jsd_split_cand

    def window_merge(_, windows, w1, w2): 
        C_other = np.vstack([C for ww, (C, _) in windows.items() if ww != w1 and ww != w2])
        C_merge = windows[w1][0] + windows[w2][0]
        return jsd(np.vstack([C_other, C_merge]))

# ============================================================================
# DESCRIPTION AND VISUALISATION

    def show(self, dims, bbs=True, nums=False, colour_class=None, ax=None, lw=0.5, alpha=1, zorder=-1): 
        dims = list(dims)
        bb_global = self.bb_global
        if ax is None: _, ax = plt.subplots()
        if colour_class is not None: col = self.C[:,:,colour_class].sum(axis=1); col = col / col.max(); cmap = cm.Greens
        for i, leaf in enumerate(self.leaves): 
            leaf_bb = leaf.bb[dims]
            leaf_bb[:,0] = np.maximum(leaf_bb[:,0], bb_global[dims,0])
            leaf_bb[:,1] = np.minimum(leaf_bb[:,1], bb_global[dims,1])
            if bbs: ax.add_patch(bb_rect(leaf_bb, fill_colour=cmap(col[i]) if colour_class is not None else None, lw=lw, alpha=alpha, zorder=zorder), )
            x, y = (leaf_bb[:,0] + leaf_bb[:,1]) / 2
            if nums: ax.text(x, y, i, horizontalalignment="center", verticalalignment="center")
        ax.set_xlabel(self.dim_names[dims[0]])
        ax.set_ylabel(self.dim_names[dims[1]])
        ax.autoscale_view()
        return ax

    def diagram(self, out_name, sf=3, verbose=False, leaf_colour="#cccccc", decision_node_colour="white", include_transitions=False):
            """
            Represent tree as a pydot diagram with pred_dims as the consequent.
            """
            idx2n = dict()
            graph_spec = 'digraph Tree {node [shape=box];'
            def _recurse(node, graph_spec, n=0, n_parent=0, dir_label="<"):
                if node.split_dim is not None:
                    split = f'{self.dim_names[node.split_dim]}={round_sf(node.split_threshold, sf)}'
                graph_spec += f'{n} [label="'
                if node.split_dim is None or verbose: 
                    if node.split_dim is None: graph_spec += f'{node.num+1}", style=filled, fillcolor="{leaf_colour}'; idx2n[node.num] = n # Leaf index.
                    # # Mean, standard deviation, range (from bb_min)
                    # if pred_dims:
                    #     for d, (mean, std, rng) in enumerate(zip(node.mean[pred_dims], np.sqrt(np.diag(node.cov)[pred_dims]), node.bb_min[pred_dims])):
                    #         graph_spec += f'{dim_names[pred_dims[d]]}: {round_sf(mean, sf)} (s={round_sf(std, sf)},r={round_sf(rng, sf)})\n'
                    # # Num samples and impurity
                    # ns = node.num_samples; graph_spec += f'n={ns}'
                    # if pred_dims: 
                    #     imp = f"{np.dot(node.var_sum[pred_dims], tree.space.global_var_scale[pred_dims]):.2E}"
                    #     graph_spec += f'\nimpurity: {imp}'
                    # if node.split_dim is not None:
                    #     graph_spec += f'\n-----\nsplit: {split}", style=filled, fillcolor="{decision_node_colour}' # Decision node (verbose)
                else: graph_spec += f'{split}", style=filled, fillcolor="{decision_node_colour}' # Decision node (non-verbose)
                graph_spec += '", fontname = "sans-serif"];'
                n_here = n
                if n_here > 0: graph_spec += f'{n_parent} -> {n} [label="{dir_label}"];' # Edge.
                n += 1
                if node.split_dim is not None: # Recurse to children.
                    graph_spec, n = _recurse(node.left, graph_spec, n, n_here, "<")
                    graph_spec, n = _recurse(node.right, graph_spec, n, n_here, ">=")
                return graph_spec, n
            graph_spec, _ = _recurse(self.root, graph_spec)
            if include_transitions:
                # Add edges for transitions - easily gets very messy!
                T = self.T_ns
                for i in range(self.m):
                    for j in range(self.m):
                        if T[i,j,0] > 0.05:
                            graph_spec += f'{idx2n[i]}:w -> {idx2n[j]}:w [label="{round_sf(T[i,j,0], sf)}"];'
            # Create and save pydot graph.    
            (graph,) = pydot.graph_from_dot_data(graph_spec+'}') 
            if out_name[-4:] == ".png":   graph.write_png(out_name) 
            elif out_name[-4:] == ".svg": graph.write_svg(out_name)
            else: raise Exception("Invalid file extension.") 

    def bb_desc(self, i, sf=3, keep_all=False): 
        """Describe the bounding box for one abstract state."""
        if i >= self.m: assert i == self.m; return "terminal"
        if self.dim_names: dim_names = self.dim_names
        else: dim_names = [f"s{j}" for j in range(self.D.shape[1])]
        terms = []
        for j, (mn, mx) in enumerate(self.leaves[i].bb):
            do_mn = mn > -np.inf; do_mx = mx < np.inf
            if do_mn and do_mx:
                terms.append(f"{round_sf(mn, sf)} =< {dim_names[j]} < {round_sf(mx, sf)}")
            elif do_mn or do_mx:
                if do_mn: terms.append(f"{dim_names[j]} >= {round_sf(mn, sf)}")
                if do_mx: terms.append(f"{dim_names[j]} < {round_sf(mx, sf)}")
            elif keep_all: terms.append("any")
        return " and ".join(terms)

    def transition_desc(self, i1, i2, sf=3, refactor=True):
        """Describe a transition via the source and destination bounding boxes."""
        b2 = self.bb_desc(i2, sf)
        if b2 == "terminal": return f"({self.bb_desc(i1, sf, keep_all=True)}) to {b2}"
        if not refactor: return f"({self.bb_desc(i1, sf, keep_all=True)}) to ({b2})"
        diff_terms, same_terms = [], []
        for b1, b2 in zip(self.bb_desc(i1, sf, keep_all=True).split(" and "), self.bb_desc(i2, sf, keep_all=True).split(" and ")):
            if b1 == b2:
                if b1 != "any": same_terms.append(b1)
            else: diff_terms.append(f"({b1} to {b2})")
        return " and ".join(diff_terms) + (" while " + " and ".join(same_terms) if same_terms else "")

    def prob_change_desc(self, i1, p1, p2, i2=None, sf=3, refactor=True):
        if i2 is not None: desc, prob1, prob2 = self.transition_desc(i1, i2, sf, refactor), self.P[i1,i2,p1],   self.P[i1,i2,p2]
        else:              desc, prob1, prob2 = self.bb_desc(i1, sf),                       self.P[i1,:,p1].sum(), self.P[i1,:,p2].sum()
        return desc + f": {round_sf(prob1, sf)} to {round_sf(prob2, sf)}"

# ============================================================================
# EXPORT

    def dict(self):
        """Create a dictionary representation of the abstraction for export."""
        C, P, T = self.C, self.P, self.T
        d = {"directed": True, "multigraph": False, "graph": {}, "var_names": self.dim_names, "classes": self.classes, "nodes": [], "links": []}
        for i, leaf in enumerate(self.leaves):
            d["nodes"].append({
                "mean": self.D[list(self.leaf2idx[i])].mean(axis=0).tolist(), 
                "bb": leaf.bb.tolist()
                })
            for j in range(self.m+1):
                if C[i,j].sum() > 0: # Don't create edges with no transitions.
                    d["links"].append({
                        "source": i,
                        "target": j,
                        "C": C[i,j].tolist(),
                        # "P": P[i,j].tolist(),
                        # "T": T[i,j].tolist()
                    })
        return d

    def dump(self, filename): dump(self, filename)

# ============================================================================

class Node: 
    def __init__(self, num, bb): 
        self.num, self.bb, self.left, self.right, self.split_dim, self.split_threshold = num, bb, None, None, None, None
    def split(self, dim, threshold, right_num): 
        self.left, self.right = Node(self.num, self.bb.copy()), Node(right_num, self.bb.copy())
        self.left.bb[dim,1] = self.right.bb[dim,0] = threshold
        self.num, self.split_dim, self.split_threshold = None, dim, threshold
        return self.left, self.right

def count(transitions, class_data, classes=None, class_mode=0, kernel=None, mask=None, min_idx=0):
    """Assemble per-class counts from transition matrix."""
    if classes is None: return np.array([[[len(t)] for t in r] for r in transitions])
    m = len(transitions); n = len(classes); C = np.zeros((m,m+1,n), dtype=(float if class_mode in {2,3} else int))
    for i, r in enumerate(transitions):
        for j, t in enumerate(r):
            if t == [] or (mask is not None and not mask[i][j]): continue # Don't look at entries that are masked out.                
            if class_mode == 1: # Range mode.
                # NOTE: Assumes both t and classes are sorted!
                a, b = 0, 0 # Range of active_classes.
                for idx in t:
                    if idx >= min_idx:
                        k = class_data[idx]
                        for kk in range(b,n): # Activate.
                            if k < classes[kk][0]: break # Window is a right-open interval.
                            b += 1                
                        for kk in range(a,b): # Deactivate.
                            if k < classes[kk][1]: break
                            a += 1
                        C[i,j,a:b] += 1
            elif class_mode == 2: # Partial membership mode.
                C[i,j] = sum(class_data[idx] for idx in t)
            elif class_mode == 3: # Kernel mode.
                if min_idx > 0: raise NotImplementedError()
                pw_sq_dist = cdist(class_data[t].reshape(-1,1), classes, "sqeuclidean")
                gaussian = np.exp(-pw_sq_dist / kernel**2) # Gaussian kernel.
                C[i,j] = gaussian.sum(axis=0)
            else: # Basic mode.
                for idx in t: 
                    if idx >= min_idx: C[i,j,classes[class_data[idx]]] += 1
    return C

def quantile_thresholds(D, q=1):
    return [np.unique(np.percentile(np.unique(D[:,d]), q=np.arange(q,100,q), interpolation="midpoint")) for d in range(D.shape[1])]

def fundamental(T): 
    Q = np.zeros_like(T)
    Q[:,:,:-1] = T[:,:,:-1]    
    return np.array([np.linalg.inv(np.identity(q.shape[0]) - q) for q in Q])

def visitation(T): return fundamental(T)[:,-1,:-1] 

def rollout(T):
    """Parallel rollout of n Markov models, yielding a joint transition distribution for each timestep."""
    # m, n = T.shape[0]-1, T.shape[2]
    # P_t = np.zeros((num_timesteps, m, m+1, n)) 
    # P_t[0,0,:m] = T[-1,:m] # Can place initial probabilities in any row so choose first arbitrarily.
    # for t in range(1, num_timesteps): P_t[t] = P_t[t-1].sum(axis=0)[:-1,None,:] * T[:-1]
    # P_sum = P_t[1:].sum(axis=0) # Real timesteps start at 1st index.
    # P = P_sum / P_sum.sum(axis=(0,1))
    # return P_t[1:], P
    """Using fundamental matrix."""
    C = visitation(T)[:,:,None] * T[:,:-1] # Counts matrix.
    return C # / C.sum(axis=(1,2))[:,None,None] # Joint probability matrix.

def jsd(M, w=None, pairwise=False): 
    """
    Population-wide or pairwise Jensen-Shannon divergence for stacked distributions.
    """
    H = entropy(M, axis=1)
    if pairwise:
        raise NotImplementedError()
        assert w is None
        n = M.shape[0]
        jsd = np.zeros((n,n))
        for p in range(n):
            for q in range(p):
                jsd[p,q] = entropy(M[[p,q]].sum(axis=0)) - (H[p] + H[q]) / 2
        return jsd + jsd.T
    else: 
        if w is None: w = M.sum(axis=1) / M.sum() # Use sums as weights if not specified.
        H_mixture = entropy(np.dot(w, M / M.sum(axis=1).reshape(-1,1)))
        jsd = H_mixture - np.dot(w, H)
        if jsd < 0: # Numerical imprecision.
            # assert jsd >= -1e-16; 
            jsd = 0. 
        return jsd

def jsd_vs_bl(M): return

def shuffle_counts(M, repeat=1):
    """Randomly shuffle the counts in M while maintaining row and column sums.
    Repeat several times if applicable, and stack along dimension 2."""
    assert M.dtype == np.int64 # Only works for integer arrays.
    col_sums, row_sums, (n, m2) = M.sum(axis=0), M.sum(axis=1), M.shape
    indices_unshuffled = [i for i, s in enumerate(col_sums) for _ in range(s)]
    M_shuffled = np.empty((n, m2, repeat), dtype=int)
    for r in range(repeat):
        indices = indices_unshuffled.copy()
        np.random.shuffle(indices)
        idx_last = 0
        for p in range(n):
            idx = idx_last + row_sums[p]
            counts = Counter(indices[idx_last:idx])
            M_shuffled[p,:,r] = [counts[i] for i in range(m2)]
            idx_last = idx    
    assert (M_shuffled.sum(axis=0).T == col_sums).all() and (M_shuffled.sum(axis=1).T == row_sums).all()
    return M_shuffled

def delta_pairwise(M):
    """Pairwise subtraction along axis 2."""
    M = np.moveaxis(M, 2, 0); return np.moveaxis(M[:,None] - M, [0,1], [2,3])

def delta_successive(M):
    """Subtraction of successive slices along axis 2."""
    return M[:,:,1:] - M[:,:,:-1]

def bb_rect(bb, fill_colour=None, edge_colour="k", alpha=1, lw=0.5, zorder=-1):
    (xl, xu), (yl, yu) = bb
    fill_bool = (fill_colour != None)
    return Rectangle(xy=[xl,yl], width=xu-xl, height=yu-yl, fill=fill_bool, facecolor=fill_colour, alpha=alpha, edgecolor=edge_colour, lw=lw, zorder=zorder) 

def round_sf(X, sf):
    try: return np.format_float_positional(X, precision=sf, unique=False, fractional=False, trim='k') 
    except: return f"[{','.join(round_sf(x, sf) for x in X)}]" 