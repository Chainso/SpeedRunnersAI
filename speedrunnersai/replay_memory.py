import numpy as np

class BinarySumTree:
    """
    A binary sum tree
    """
    def __init__(self, num_leaves):
        """
        Creates a binary sum tree with the given number of leaves

        num_leaves : The number of leaves in the tree
        """
        self.num_leaves = num_leaves
        self.size = 0
        self.current_index = 0

        self.tree = np.zeros(2 * self.num_leaves - 1)

    def __len__(self):
        """
        Returns the number of values added up to the number of leaves in the
        tree
        """
        return self.size

    def _update_parents(self, index, value):
        """
        Updates all the parent nodes going up to the root to accomodate the
        addition of a new node

        index : The index of the new node
        value : The value of the new node
        """
        change_in_value = value - self.tree[index]
        parent = index

        # Keep updating until the root node is reached
        while(parent != 0):
            parent = (parent - 1) // 2
            self.tree[parent] += change_in_value

    def _leaf_start_index(self):
        """
        Returns the starting index of the leaves
        """
        return self.num_leaves - 1

    def _leaf_idx_to_real(self, leaf_index):
        """
        Converts the index of a leaf relative to the other leaves to the index
        in the tree (num_leaves - 1 + leaf_index)

        leaf_index : The index of the leaf to be convert to the tree index
        """
        return self._leaf_start_index() + leaf_index

    def add(self, value):
        """
        Pushes the given value onto the sum tree. When the tree is at capacity,
        the values will be replaced starting with the first one

        value : The value of the item
        """
        self.set(value, self.current_index)
        self.current_index += 1

        if(self.size < self.num_leaves):
            self.size += 1

        if(self.current_index == self.num_leaves):
            self.current_index = 0

    def set(self, value, index):
        """
        Sets the value of the leaf at the leaf index given

        value : The value of the leaf
        index : The index of the leaf
        """
        tree_index = self._leaf_idx_to_real(index)
        self._update_parents(tree_index, value)
        self.tree[tree_index] = value

    def get(self, index):
        """
        Retrieves the node at the given index

        index : The index of the node to retrieve
        """
        return self.tree[index]

    def get_leaf(self, leaf_index):
        """
        Returns the leaf with the given index relative to the leaves

        leaf_index : The index of the leaf relative to other leaves
        """
        tree_index = self._leaf_idx_to_real(leaf_index)
        return self.get(tree_index)

    def get_leaves(self):
        """
        Returns all the added leaves in the tree
        """
        leaf_start = self._leaf_start_index()
        return self.tree[leaf_start:leaf_start + self.size]

    def sum(self):
        """
        Returns the sum of the tree (the value of the root)
        """
        return self.tree[0]

    def next_index(self):
        """
        Returns the leaf index of the next value added
        """
        return self.current_index

class PERMemory:
    """
    A Prioritized Experience Replay implementation
    https://arxiv.org/abs/1511.05952
    """
    def __init__(self, capacity, alpha, beta, beta_increment, epsilon):
        """
        Creates a new PER buffer with the given parameters

        capacity : The capacity of the replay buffer
        alpha : The alpha value for the prioritization, between 0 and 1
                inclusive
        beta : The beta value for the importance sampling, between 0 and 1
               inclusive
        beta_increment : The value to increment the beta by
        epsilon : The value of epsilon to add to the priority
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.experiences = np.zeros(capacity, dtype=object)
        self.priorities = BinarySumTree(capacity)

    def __len__(self):
        """
        Returns the number of experiences added to the buffer
        """
        return len(self.priorities)

    def _get_priority(self, error):
        """
        Computes the priority for the given error

        error : The error to get the priority for
        """
        return (error + self.epsilon) ** self.alpha

    def add(self, experience, error):
        """
        Adds the given experience to the replay buffer with the priority being
        the given error added to the epsilon value

        experience : The experience to add to the buffer
        error : The TD-error of the experience
        """
        current_index = self.priorities.next_index()
        self.experiences[current_index] = np.array(experience)

        priority = self._get_priority(error)
        self.priorities.add(priority)

    def sample(self, size):
        """
        Samples "size" number of experiences from the buffer

        size : The number of experiences to sample
        """
        priorities = self.priorities.get_leaves() / self.priorities.sum()
        indices = np.random.choice(len(priorities), size, p = priorities)

        batch = np.stack(self.experiences[indices])

        stacked_batch = []

        # In order to get float32 instead of float64 and long over int
        for arr in batch.transpose():
            stacked_arr = np.stack(arr)

            if(stacked_arr.dtype == np.float64):
                stacked_arr = stacked_arr.astype(np.float32)
            elif(stacked_arr.dtype == np.int32):
                stacked_arr = stacked_arr.astype(np.int64)

            stacked_batch.append(stacked_arr)

        probabilities = priorities[indices]

        is_weights = np.power(len(self.priorities) * probabilities, -self.beta)
        is_weights /= is_weights.max()

        self.beta = np.min([1.0, self.beta + self.beta_increment])

        return stacked_batch, indices, is_weights

    def update_priority(self, index, error):
        """
        Updates the priority of the experience at the given index, using the
        error given

        index : The index of the experience
        error : The new error of the experience
        """
        priority = self._get_priority(error)
        self.priorities.set(priority, index)

    def update_priorities(self, indices, errors):
        """
        Updates the priority of the experiences at the given indices, using the
        errors given

        index : The index of the experience
        error : The new error of the experience
        """
        for index, error in zip(indices, errors):
            self.update_priority(index, error)
