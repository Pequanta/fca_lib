from typing import Dict, List, Set, Tuple
from collections import Counter
from fca.utils.bitset import BitSetOperations
from fca.concept_lattice import ConceptLattice
from fca.algorithms.iceberg_concept import IcebergConcept
from fca.qubo_formulation.qubo_formulations import QuboFormulation
from fca.qubo_formulation.classical_solutions import ClassicalSolutions
from scripts.examples.jsm_method.data_preprocessing import PreprocessingJSM
from fca.qubo_formulation.dirac_solution import DiracSolution
import numpy as np
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

bset_operations = BitSetOperations()
iceberg_operations = IcebergConcept()
qubo_formulation = QuboFormulation()

class JSMMethodApplication:
    def __init__(
            self,
            goal_attr: List, 
            objects_: List,
            attributes_: List,
            data_preprocessing: PreprocessingJSM,
            solution_type: str,
            n_rules: int = 30,
            min_support: float = 0.176
            ):
        self.min_support = min_support
        self.positive_hypotheses = []
        self.negative_hypotheses = []
        self.goal_attr = goal_attr
        self.objects_ = objects_
        self.attributes_ = attributes_
        self.data_processing = data_preprocessing
        self.positive_context, self.negative_context = data_preprocessing.process_data() 
        self.n_rules = n_rules

        self.candidates_data = None
        self.qubo_data = None


        #positive concepts
        ext_int, int_ext, _, _ = self.data_processing.encode_data(1)
        positive_lattice = ConceptLattice(ext_int, int_ext, self.objects_[:len(self.positive_context)], self.attributes_, self.min_support)
  
        self.positive_concepts = positive_lattice.all_concepts()
        self.baseline_counts = {}

        self.solution_type = solution_type


        #calculating the baseline counts, i.e,  the number of occurrences of each class for all training examples
        self.baseline_counts = Counter({0: 0, 1: 0})
        index = 0
        while index < len(self.goal_attr):
            if self.goal_attr[index] == 1:
                self.baseline_counts[1] += 1
            else:
                self.baseline_counts[0] += 1
            index += 1


    def train(self):
        """
            Generates hypotheses and candidates, and solves the QUBO problem to select candidate concepts.
        """
        self.positive_hypotheses = self.get_hypotheses(type=True)
        self.negative_hypotheses = self.get_hypotheses(type=False)
        self.candidates_data = self.get_candidates()
        self.candidate_concepts = list(self.candidates_data.keys())
        self.qubo_data = self.get_qubo_data(self.candidates_data, self.candidate_concepts, self.positive_hypotheses, alpha=self.min_support, beta=1.0, n_rules=self.n_rules, solution_type=self.solution_type)
        
        print("Candidate size: ", len(self.candidate_concepts))

        
       

    def get_hypotheses(self, type: int, undetermined_context = None) -> List[int]:
        """

            This method can be used to find all valid hypotheses from the positive and negative contexts or
            it can be used to find the valid hypotheses for which the undetermined context is a subset of.


            Args:
                type(bool): a boolean value indicating whether to return positive or negative hypotheses
                undetermined_context: list of contexts from the test data
            Returns:
                a set containing all of possible hypotheses of the indicated type
        """

        candidate_hyp = []
        ext_int, int_ext, _, _ = self.data_processing.encode_data(type)
        lattice = ConceptLattice(ext_int, int_ext, self.objects_[:len(self.positive_context)], self.attributes_, self.min_support)
        concepts = lattice.all_concepts()
        for i in range(len(concepts)):
            if undetermined_context is not None and bset_operations.__subset_of__(undetermined_context, concepts[i][0]): # type: ignore
                if self.check_valid_hypothesis(undetermined_context, type=1):
                    candidate_hyp.append(concepts[i][1])
            else:
                if type == True and self.check_valid_hypothesis(concepts[i][1], type=1): # type: ignore
                    candidate_hyp.append(concepts[i][1])
                elif type == False and self.check_valid_hypothesis(concepts[i][1], type=-1): # type: ignore
                    candidate_hyp.append(concepts[i][1])

        return candidate_hyp
    

    def check_valid_hypothesis(self, hypothesis: int, type: int) -> bool:
        """
            Checks if a hypothesis is valid by ensuring it is not a subset of the opposite context.

            Args:
                hypothesis (int): Bitset representation of the hypothesis.
                type (int): 1 for positive, -1 for negative.

            Returns:
                bool: True if valid, False otherwise.
        """
        is_valid = True
        length = len(self.negative_context) if type==1 else len(self.positive_context)
        for i in range(length):
            if type == 1:
                #Checking if the hypothesis is a subset of the negative context
                #If it is a subset then it is not a valid hypothesis
                if not bset_operations.__subset_of__(hypothesis, self.negative_context[i]):
                    is_valid = False
                    break
            else:
                #Checking if the hypothesis is a subset of the positive context
                #If it is a subset then it is not a valid hypothesis
                if not bset_operations.__subset_of__(hypothesis, self.positive_context[i]):
                    is_valid = False
                    break
        return is_valid

    def get_class_counts(self, undetermined_context):
        """
            Counts the occurrences of each class label in the positive context that match the undetermined context.

            Args:
                undetermined_context (int): Bitset representation of the test instance's attributes.

            Returns:
                Dict[int, int]: Dictionary with counts for each class label.
        """
        result = {0: 0, 1: 0}
        for i in range(len(self.positive_context)):
            if bset_operations.__subset_of__(undetermined_context, self.positive_context[i]):
                result[self.goal_attr[i]] += 1
        return result

    def get_candidates(self):
        """
            Generates candidate concepts and their iceberg data from positive concepts.

            Args:
                None

            Returns:
                Dict: Dictionary mapping concepts to their iceberg data.
        """

        # Getting iceberg data for positive concepts
        candidates_data = {
            self.positive_concepts[i]: iceberg_operations.get_iceberg_data(
                self.positive_concepts[i],
                self.goal_attr,
                len(self.objects_),
                self.baseline_counts
            )
            for i in range(len(self.positive_concepts)) if self.positive_concepts[i][0] != 0
        }


        return candidates_data

    def get_qubo_data(self, candidates, candidate_concepts, context, alpha, beta,n_rules, solution_type = "classical") -> List[Dict] | None:
        """
        Generates QUBO data and selects candidates based on the specified solution type.

        Args:
            candidates (List): List of candidate items to be evaluated.
            candidate_concepts (List): List of concepts corresponding to each candidate.
            context (Any): Contextual information required for QUBO formulation.
            alpha (float): Weight parameter for the QUBO formulation.
            beta (float): Weight parameter for the QUBO formulation.
            n_rules (int): Number of rules to be considered in the QUBO formulation.
            solution_type (str, optional): Type of solution to use ("classical" or "quantum"). Defaults to "classical".

        Returns:
            List[Dict] | None: List of selected candidate concepts if a valid solution type is specified, otherwise None.
        """
        Q_matrix, offset = qubo_formulation.build_qubo(candidates, context, alpha, beta, n_rules)
        if solution_type == "classical":
            solution_ = ClassicalSolutions(Q_matrix)
            sel_vec, energy = solution_.solve(offset=offset, n_iters=2000, temp_start=1.0, temp_end=1e-3, seed=42)
        elif solution_type == "quantum":
            solution_ = DiracSolution(Q_matrix)
            sel_vec, energy = solution_.solve() #type: ignore
        else:
            print("Solution type not specified")
            return None
        selected_candidates = [candidate_concepts[i] for i in range(len(candidate_concepts)) if sel_vec[i] == 1]
        print("Q matrix size: ", Q_matrix.shape)
        return selected_candidates

    def classify_(self, undetermined_context: int, index: int) -> Dict[int, float] | int | None:
        """
            Classifies a single undetermined context using the trained JSM method and QUBO-selected candidates.

            Args:
                undetermined_context (int): Bitset representation of the test instance's attributes.
                index (int): Index of the test instance (not used in logic, but may be for logging or future extensions).

            Returns:
                int: Predicted class label (0 or 1) for the test instance.
                None: If no candidates or QUBO data are available.
        """
    

        #classification based on the data
        if len(self.candidates_data) == 0: #type: ignore
            return self.baseline_counts.most_common(1)[0][0] # type: ignore
        # Get QUBO data
       
        if len(self.qubo_data) == 0:#type: ignore
            return self.baseline_counts.most_common(1)[0][0] # type: ignore
        votes = Counter({0: 0, 1: 0})
        candidate_intents = [self.candidate_concepts[i][1] for i in range(len(self.qubo_data)) if len(self.qubo_data) > 0 and self.qubo_data[i][1] > 0] #type: ignore
        for i in range(len(candidate_intents)):
            if bset_operations.__subset_of__(candidate_intents[i], undetermined_context):#type: ignore
                votes[1] += 1
            else:
                votes[0] += 1

        return votes.most_common(1)[0][0] if len(votes) > 0 else self.baseline_counts.most_common(1)[0][0] # type: ignore

    def get_accuracy(self, predictions: int, testing_target: int, data_size: int):
        """
            !!!!all binary representations are read from right to left where the right most bit representing index=0
            Args:
                predictions: target values predicted by the classification task interms of binary representation of
                             an integer where 1 represents the predictions of True and 0 represents False
                testing_target: target values supplied by the dataset interms of binary representation of an integer
                                where 1 represents the true value of True of the target and 0 False.
                data_size: an integer indicating the number of testing examples
            Returns:
                result: an integer whose binary values represent whether the predicted values are correct or wrong.
                        1 representing correctly predicted values while 0 representing falsely represented values.
        """

        #Get correctly labeled values
        result = (1 << data_size) - 1
        i = 0
        while data_size > 0:
            if (predictions & (1 << i)) != (testing_target & (1 << i)):
                result ^= (1 << i)
            i += 1
            data_size -= 1
        return result.bit_count() / data_size
