import copy
import random


class ValueEntry:
    """
    Represents a value entry with the number of gossiping participations 
    and the current value.
    """

    def __init__(self, participations, value):
        """
        Initialize a ValueEntry object.

        Args:
            participations (int): The number of participations.
            value (int): The value.
        """
        self.participations = int(participations)
        self.value = int(value)


class Algorithm:
    """
    Represents an algorithm with a name and a list of neighbors.
    The algorithm's function is to select a neighbor for gossiping.
    """

    def __init__(self, name, neighbors):
        """
        Initialize an Algorithm object.

        Args:
            name (str): The name of the algorithm.
            neighbors (list): The list of neighbors.
        """
        self.name = name
        self.neighbors = neighbors
        print(f"Running algorithm: {self.name}.")
        print(f"Received neighbors: {self.neighbors}.")
        self.modifiable_parameters = False
        self.weights = {}

    def select_neighbor(self):
        """
        Select a neighbor randomly.

        Returns:
            object: The selected neighbor.
        """
        return random.choice(self.neighbors)


class Memory:
    """
    Represents a memory object with memory storage and prior partner factor.
    The factor is used to discourage repeated gossiping between the same nodes.
    """

    def __init__(self):
        self.modifiable_parameters = None
        self.prior_partner_factor = None
        self.prior_partner_factor_index = None
        self.prior_partner_factors = None
        self.memory = None

    def init_memory(self, prior_partner_factors):
        """
        Initialize the memory.

        Args:
            prior_partner_factors (list): The prior partner factors.
        """
        self.memory = set()
        self.prior_partner_factors = prior_partner_factors
        self.prior_partner_factor_index = 0
        self.prior_partner_factor = self.prior_partner_factors[self.prior_partner_factor_index]
        self.modifiable_parameters = True
        print(f'Initialized Memory with prior partner factor {self.prior_partner_factor}.')

    def remember(self, neighbor):
        """
        Remember a neighbor.

        Args:
            neighbor: The neighbor to remember.
        """
        if neighbor not in self.memory:
            self.weights[neighbor] *= self.prior_partner_factor
            self.memory.add(neighbor)

    def reset_memory(self):
        """
        Reset the memory to the start config.
        """
        print('Memory reset.')
        self.memory = set()

    def set_next_memory_parameters(self):
        """
        Sets the next memory parameters from the list.
        """
        self.prior_partner_factor_index += 1
        self.prior_partner_factor = self.prior_partner_factors[self.prior_partner_factor_index]
        print(f'Set new prior partner factor {self.prior_partner_factor}.')


class ComplexMemory(Memory):
    """
    Represents a complex memory object that inherits from Memory.
    Each interaction with a neighbor further penalizes the probability for selection.
    It also enhances the normal memory with the ability to forget.
    Forgetting restores the start weights.
    """

    def __init__(self):
        super().__init__()
        self.start_weights = None

    def init_memory(self, prior_partner_factors):
        """
        Initialize the complex memory.

        Args:
            prior_partner_factors (list): The prior partner factors.
        """
        self.prior_partner_factors = prior_partner_factors
        self.prior_partner_factor_index = 0
        self.prior_partner_factor = self.prior_partner_factors[self.prior_partner_factor_index]
        self.start_weights = copy.deepcopy(self.weights)
        self.modifiable_parameters = True
        print(f'Initialized ComplexMemory with prior partner factor {self.prior_partner_factor}.')

    def remember(self, neighbor):
        """
        Remember a neighbor and update weights.

        Args:
            neighbor: The neighbor to remember.
        """
        self.weights[neighbor] *= self.prior_partner_factor

    def forget(self):
        """
        Forget previously remembered neighbors and reset weights.
        """
        self.weights = copy.deepcopy(self.start_weights)

    def reset_memory(self):
        """
        Reset the memory to the start config.
        """
        print('Memory reset.')
        self.forget()


class DefaultMemory(Algorithm, Memory):
    """
    Represents the default memory algorithm. It inherits from both Algorithm and Memory.
    """

    def __init__(self, name, neighbors, prior_partner_factors):
        """
        Initialize the default memory.

        Args:
            name (str): The name of the memory.
            neighbors (list): The list of neighbors.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors)
        self.weights = {}
        for neighbor in self.neighbors:
            self.weights[neighbor] = 1.0
        self.init_memory(prior_partner_factors)

    def select_neighbor(self):
        """
        Select a neighbor based on weights.

        Returns:
            object: The selected neighbor.
        """
        selected = random.choices(self.neighbors, weights=self.weights.values())[0]
        return selected

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        self.set_next_memory_parameters()


class DefaultComplexMemory(Algorithm, ComplexMemory):
    """
    Represents the default complex memory algorithm. It inherits from both Algorithm and ComplexMemory.
    """

    def __init__(self, name, neighbors, prior_partner_factors):
        """
        Initialize the default complex memory.

        Args:
            name (str): The name of the memory.
            neighbors (list): The list of neighbors.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors)
        self.weights = {}
        for neighbor in self.neighbors:
            self.weights[neighbor] = 1.0
        self.init_memory(prior_partner_factors)

    def select_neighbor(self):
        """
        Select a neighbor based on weights.

        Returns:
            object: The selected neighbor.
        """
        selected = random.choices(self.neighbors, weights=self.weights.values())[0]
        return selected

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        self.set_next_memory_parameters()


class WeightedFactor(Algorithm):
    """
    Represents a weighted factor object that inherits from Algorithm.
    """

    def __init__(self, name, neighbors, community_neighbors, factors):
        """
        Initialize the weighted factor.

        Args:
            name (str): The name of the weighted factor.
            neighbors (list): The list of neighbors.
            community_neighbors (list): The list of community neighbors.
            factors (list): The factor to prioritize non-community neighbors.
        """
        super().__init__(name, neighbors)
        self.community_neighbors = community_neighbors
        self.factors = factors
        self.factor_index = 0
        self.factor = self.factors[self.factor_index]
        self.compute_weights()
        self.modifiable_parameters = True
        print(f'Initialized WeightedFactor with factor {self.factor}.')

    def compute_weights(self):
        self.weights = {}
        for neighbor in self.neighbors:
            if neighbor in self.community_neighbors:
                weight = 1.0
            else:
                # prioritize non-community neighbors with a given factor
                weight = self.factor
            self.weights[neighbor] = weight

    def select_neighbor(self):
        """
        Select a neighbor based on weights.

        Returns:
            object: The selected neighbor.
        """
        selected = random.choices(self.neighbors, weights=self.weights.values())[0]
        return selected

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        self.factor_index += 1
        self.factor = self.factors[self.factor_index]
        print(f'Set new factor {self.factor}.')
        self.compute_weights()


class WeightedFactorMemory(WeightedFactor, Memory):
    """
    Represents the weighted factor memory algorithm that inherits from both WeightedFactor and Memory.
    """

    def __init__(self, name, neighbors, community_neighbors, factors, prior_partner_factors):
        """
        Initialize the weighted factor memory.

        Args:
            name (str): The name of the memory.
            neighbors (list): The list of neighbors.
            community_neighbors (list): The list of community neighbors.
            factors (list): The factors to prioritize non-community neighbors.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors, community_neighbors, factors)
        self.init_memory(prior_partner_factors)

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        self.set_next_memory_parameters()
        super().set_next_parameters()


class WeightedFactorComplexMemory(WeightedFactor, ComplexMemory):
    """
    Represents the weighted factor complex memory algorithm that inherits from both WeightedFactor and ComplexMemory.
    """

    def __init__(self, name, neighbors, community_neighbors, factors, prior_partner_factors):
        """
        Initialize the weighted factor complex memory.

        Args:
            name (str): The name of the memory.
            neighbors (list): The list of neighbors.
            community_neighbors (list): The list of community neighbors.
            factors (list): The factors to prioritize non-community neighbors.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors, community_neighbors, factors)
        self.init_memory(prior_partner_factors)

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        self.set_next_memory_parameters()
        super().set_next_parameters()


class CommunityProbabilities(Algorithm):
    """
    Represents a community probabilities object that inherits from Algorithm.
    """

    def __init__(self, name, neighbors, same_community_probabilities_neighbors):
        """
        Initialize the community probabilities.

        Args:
            name (str): The name of the community probabilities.
            neighbors (list): The list of neighbors.
            same_community_probabilities_neighbors (list): The list of same community probabilities for neighbors.
        """
        super().__init__(name, neighbors)
        self.same_community_probabilities_neighbors = same_community_probabilities_neighbors
        inverted_probabilities = [1 / x for x in same_community_probabilities_neighbors]
        # set weights to inverted probabilities
        self.weights = {}
        for i in range(0, len(self.neighbors)):
            self.weights[neighbors[i]] = inverted_probabilities[i]

    def select_neighbor(self):
        """
        Select a neighbor based on weights.

        Returns:
            object: The selected neighbor.
        """
        selected = random.choices(self.neighbors, weights=self.weights.values())[0]
        return selected


class CommunityProbabilitiesMemory(CommunityProbabilities, Memory):
    """
    Represents a community probabilities memory algorithm that inherits from both CommunityProbabilities and Memory.
    """

    def __init__(self, name, neighbors, same_community_probabilities_neighbors, prior_partner_factors):
        """
        Initialize the community probabilities memory.

        Args:
            name (str): The name of the memory.
            neighbors (list): The list of neighbors.
            same_community_probabilities_neighbors (list): The list of same community probabilities for neighbors.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors, same_community_probabilities_neighbors)
        self.init_memory(prior_partner_factors)

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        self.set_next_memory_parameters()


class CommunityProbabilitiesComplexMemory(CommunityProbabilities, ComplexMemory):
    """
    Represents a community probabilities complex memory algorithm that inherits from both CommunityProbabilities and ComplexMemory.
    """

    def __init__(self, name, neighbors, same_community_probabilities_neighbors, prior_partner_factors):
        """
        Initialize the community probabilities complex memory.

        Args:
            name (str): The name of the memory.
            neighbors (list): The list of neighbors.
            same_community_probabilities_neighbors (list): The list of same community probabilities for neighbors.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors, same_community_probabilities_neighbors)
        self.init_memory(prior_partner_factors)

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        self.set_next_memory_parameters()


class WeightUpdate:
    """
    Represents the weight update component.
    """

    def update_weights(self, other_weights, a, b):
        """
        Update weights based on a combination of self weights and other weights.

        Args:
            other_weights (list): List of weights from other source.
            a (float): Coefficient for self weights.
            b (float): Coefficient for other weights.
        """
        updated_weights = {}
        for idx, n in enumerate(self.neighbors):
            updated_weights[n] = a * self.weights[n] + b * other_weights[idx]
        self.weights = updated_weights


class AdvancedWeightedFactor(WeightedFactor, WeightUpdate):
    """
    Represents the combination of advanced centrality and weighted factor algorithm.
    Inherits from both WeightedFactor and WeightUpdate.
    """

    def __init__(self, name, neighbors, community_neighbors, factors,
                 advanced_weights_neighbors, params_a, params_b):
        """
        Initialize AdvancedWeightedFactor instance.

        Args:
            name (str): The name of the instance.
            neighbors (list): List of neighbors.
            community_neighbors (list): List of community neighbors.
            factors (list): The factor to prioritize non-community neighbors.
            advanced_weights_neighbors (list): List of advanced weights for neighbors.
            params_a (list): Coefficients for self weights.
            params_b (list): Coefficients for advanced weights.
        """
        super().__init__(name, neighbors, community_neighbors, factors)
        self.advanced_weights_neighbors = advanced_weights_neighbors
        self.params_a = params_a
        self.params_b = params_b
        self.update_weights(self.advanced_weights_neighbors,
                            self.params_a[self.factor_index],
                            self.params_b[self.factor_index])

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        super().set_next_parameters()
        self.update_weights(self.advanced_weights_neighbors,
                            self.params_a[self.factor_index],
                            self.params_b[self.factor_index])


class AdvancedWeightedFactorMemory(WeightedFactorMemory, WeightUpdate):
    """
    Represents an advanced centrality and weighted factor memory algorithm
    that inherits from both AdvancedWeightedFactor and Memory.
    """

    def __init__(self, name, neighbors, community_neighbors, factors,
                 advanced_weights_neighbors, params_a, params_b, prior_partner_factors):
        """
        Initialize the advanced centrality and weighted factor memory.

        Args:
            name (str): The name of the instance.
            neighbors (list): List of neighbors.
            community_neighbors (list): List of community neighbors.
            factors (list): The factor to prioritize non-community neighbors.
            advanced_weights_neighbors (list): List of advanced weights for neighbors.
            params_a (list): Coefficients for self weights.
            params_b (list): Coefficients for advanced weights.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors, community_neighbors, factors, prior_partner_factors)
        self.advanced_weights_neighbors = advanced_weights_neighbors
        self.params_a = params_a
        self.params_b = params_b
        self.update_weights(self.advanced_weights_neighbors,
                            self.params_a[self.factor_index],
                            self.params_b[self.factor_index])

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        super().set_next_parameters()
        self.update_weights(self.advanced_weights_neighbors,
                            self.params_a[self.factor_index],
                            self.params_b[self.factor_index])


class AdvancedWeightedFactorComplexMemory(WeightedFactorComplexMemory, WeightUpdate):
    """
    Represents an advanced centrality and weighted factor complex memory algorithm
    that inherits from both AdvancedWeightedFactor and ComplexMemory.
    """

    def __init__(self, name, neighbors, community_neighbors, factors,
                 advanced_weights_neighbors, params_a, params_b, prior_partner_factors):
        """
        Initialize the advanced centrality and weighted factor complex memory.

        Args:
            name (str): The name of the instance.
            neighbors (list): List of neighbors.
            community_neighbors (list): List of community neighbors.
            factors (list): The factor to prioritize non-community neighbors.
            advanced_weights_neighbors (list): List of advanced weights for neighbors.
            params_a (list): Coefficients for self weights.
            params_b (list): Coefficients for advanced weights.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors, community_neighbors, factors, prior_partner_factors)
        self.advanced_weights_neighbors = advanced_weights_neighbors
        self.params_a = params_a
        self.params_b = params_b
        self.update_weights(self.advanced_weights_neighbors,
                            self.params_a[self.factor_index],
                            self.params_b[self.factor_index])

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        super().set_next_parameters()
        self.update_weights(self.advanced_weights_neighbors,
                            self.params_a[self.factor_index],
                            self.params_b[self.factor_index])


class AdvancedCommunityProbabilities(CommunityProbabilities, WeightUpdate):
    """
    Represents the advanced centrality and community probability algorithm.
    """

    def __init__(self, name, neighbors, same_community_probabilities_neighbors,
                 advanced_weights_neighbors, params_a, params_b):
        """
        Initialize AdvancedCommunityProbabilities instance.

        Args:
            name (str): The name of the instance.
            neighbors (list): List of neighbors.
            same_community_probabilities_neighbors (list): List of same community probabilities for neighbors.
            advanced_weights_neighbors (list): List of advanced weights for neighbors.
            params_a (list): Coefficients for self weights.
            params_b (list): Coefficients for advanced weights.
        """
        super().__init__(name, neighbors, same_community_probabilities_neighbors)
        self.params_a = params_a
        self.params_b = params_b
        self.factor_index = 0
        self.update_weights(self.advanced_weights_neighbors,
                            self.params_a[self.factor_index],
                            self.params_b[self.factor_index])

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        self.factor_index += 1
        self.update_weights(self.advanced_weights_neighbors,
                            self.params_a[self.factor_index],
                            self.params_b[self.factor_index])


class AdvancedCommunityProbabilitiesMemory(CommunityProbabilitiesMemory, WeightUpdate):
    """
    Represents an advanced centrality and community probabilities memory algorithm that inherits from both
    AdvancedCommunityProbabilities and Memory.
    """

    def __init__(self, name, neighbors, same_community_probabilities_neighbors,
                 advanced_weights_neighbors, params_a, params_b, prior_partner_factors):
        """
        Initialize the advanced centrality and community probabilities memory.

        Args:
            name (str): The name of the memory.
            neighbors (list): List of neighbors.
            same_community_probabilities_neighbors (list): List of same community probabilities for neighbors.
            advanced_weights_neighbors (list): List of advanced weights for neighbors.
            params_a (list): Coefficients for self weights.
            params_b (list): Coefficients for advanced weights.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors, same_community_probabilities_neighbors, prior_partner_factors)
        self.advanced_weights_neighbors = advanced_weights_neighbors
        self.params_a = params_a
        self.params_b = params_b
        self.factor_index = 0
        self.update_weights(self.advanced_weights_neighbors,
                            self.params_a[self.factor_index],
                            self.params_b[self.factor_index])

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        self.factor_index += 1
        super().set_next_parameters()
        self.update_weights(self.advanced_weights_neighbors,
                            self.params_a[self.factor_index],
                            self.params_b[self.factor_index])


class AdvancedCommunityProbabilitiesComplexMemory(CommunityProbabilitiesComplexMemory, WeightUpdate):
    """
    Represents an advanced centrality and community probabilities complex memory algorithm that inherits from both
    AdvancedCommunityProbabilities and ComplexMemory.
    """

    def __init__(self, name, neighbors, same_community_probabilities_neighbors,
                 advanced_weights_neighbors, params_a, params_b, prior_partner_factors):
        """
        Initialize the advanced centrality and community probabilities complex memory.

        Args:
            name (str): The name of the memory.
            neighbors (list): List of neighbors.
            same_community_probabilities_neighbors (list): List of same community probabilities for neighbors.
            advanced_weights_neighbors (list): List of advanced weights for neighbors.
            params_a (list): Coefficients for self weights.
            params_b (list): Coefficients for advanced weights.
            prior_partner_factors (list): The prior partner factors.
        """
        super().__init__(name, neighbors, same_community_probabilities_neighbors, prior_partner_factors)
        self.advanced_weights_neighbors = advanced_weights_neighbors
        self.params_a = params_a
        self.params_b = params_b
        self.factor_index = 0
        self.update_weights(self.advanced_weights_neighbors,
                            self.params_a[self.factor_index],
                            self.params_b[self.factor_index])

    def set_next_parameters(self):
        """
        Sets the next parameters from the list.
        """
        self.factor_index += 1
        super().set_next_parameters()
        self.update_weights(self.advanced_weights_neighbors,
                            self.params_a[self.factor_index],
                            self.params_b[self.factor_index])


class CommunityBased(Algorithm):
    """
    Represents the community based algorithm from related work object that inherits from Algorithm.
    """

    def __init__(self, name, neighbors, neighboring_communities):
        """
        Initialize the community based.

        Args:
            name (str): The name of the weighted factor.
            neighbors (list): The list of neighbors.
            neighboring_communities (list): The list of neighboring communities.
        """
        super().__init__(name, neighbors)
        self.neighboring_communities = neighboring_communities
        # Calculate the community members count
        self.community_members_count_dict = {}
        for comm in self.neighboring_communities:
            if comm not in self.community_members_count_dict:
                self.community_members_count_dict[comm] = 1
            else:
                self.community_members_count_dict[comm] += 1

        # Calculate the number of distinct communities
        self.comm_count = len(self.community_members_count_dict)
        self.compute_weights()
        print(f'Initialized CommunityBased.')

    def compute_weights(self):
        self.weights = {}
        for index, neighbor in enumerate(self.neighbors):
            neighbor_community = self.neighboring_communities[index]
            # Calculate the weight based on the formula
            weight = (1 / self.comm_count) * (1 / self.community_members_count_dict[neighbor_community])
            self.weights[neighbor] = weight

name = 'betweenness_community_probabilities_complex_memory'
neighbors = ['high-mod-betweenness-cp-cm-g0-n1', 'high-mod-betweenness-cp-cm-g0-n27']
same_community_probabilities_neighbors = [0.75, 1.0]
advanced_weights_neighbors = [0.5706, 0.0]
params_a = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
params_b = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
prior_partner_factors = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

advancedCommunityProbabilitiesComplexMemory = AdvancedCommunityProbabilitiesComplexMemory(
    name, neighbors, same_community_probabilities_neighbors,
    advanced_weights_neighbors, params_a, params_b, prior_partner_factors
)
