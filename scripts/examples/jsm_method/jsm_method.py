from typing import Dict, List, Set, Tuple
from numpy import binary_repr
from collections import Counter
from fca.utils.bitset import BitSetOperations
from fca.utils.tools import count_ones
from fca.concept_lattice import ConceptLattice
from fca.algorithms.iceberg_concept import IcebergConcept
from fca.qubo_formulation.qubo_formulations import QuboFormulation
from fca.qubo_formulation.classical_solutions import ClassicalSolutions
from scripts.examples.jsm_method.data_preprocessing import PreprocessingJSM


bset_operations = BitSetOperations()
iceberg_operations = IcebergConcept()
qubo_formulation = QuboFormulation()
classical_solutions = ClassicalSolutions()
class JSMMethodApplication:
    def __init__(
            self,
            goal_attr: List, 
            objects_: List,
            attributes_: List,
            data_preprocessing: PreprocessingJSM

            ):
        self.postive_hypotheses = []
        self.negative_hypotheses = []
        self.goal_attr = goal_attr
        self.objects_ = objects_
        self.attributes_ = attributes_
        self.data_processing = data_preprocessing
        self.positive_context, self.negative_context, self.uknown_context = data_preprocessing.process_data() 
        
        
        #positive concepts
        ext_int, int_ext, _, _ = self.data_processing.encode_data(1)
        positive_lattice = ConceptLattice(ext_int, int_ext, self.objects_, self.attributes_)
        self.concepts = positive_lattice.all_concepts()


        self.baseline_counts = {}


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
            The method will handle generation of hypotheses from the positive and negative contexts.
        """
        self.postive_hypotheses = self.get_hypotheses(type=True)
        self.negative_hypotheses = self.get_hypotheses(type=False)

    def get_hypotheses(self, type: int, undetermined_context = None) -> List[int]:
        """

            This method can be used to find all valid hypotheses from the positive and negative contexts or
            it can be used to find the valid hypotheses for which the undetermined context is a subset of.


            Args:
                type(bool): a boolean value indicating whether to return positive or negative hypotheses
            Returns:
                a set containing all of possible hypotheses of the indicated type
        """

        candidate_hyp = []
        ext_int, int_ext, _, _ = self.data_processing.encode_data(type)
        lattice = ConceptLattice(ext_int, int_ext, self.objects_, self.attributes_)
        concepts = lattice.all_concepts()
        for i in range(len(concepts)):
            if undetermined_context is not None and bset_operations.__subset_of__(undetermined_context, concepts[i][0]): # type: ignore
                if self.check_valid_hypothesis(undetermined_context, type=1):
                    candidate_hyp.append(concepts[i][1])
            else:
                if type == 1 and self.check_valid_hypothesis(concepts[i][1], type=1): # type: ignore
                    candidate_hyp.append(concepts[i][1])
                elif type == -1 and self.check_valid_hypothesis(concepts[i][1], type=-1): # type: ignore
                    candidate_hyp.append(concepts[i][1])

        return candidate_hyp
    

    def check_valid_hypothesis(self, hypothesis: int, type: int) -> bool:
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
        result = {0: 0, 1: 0}

        for i in range(len(self.positive_context)):
            if bset_operations.__subset_of__(undetermined_context, self.positive_context[i]):
                result[self.goal_attr[i]] += 1
        return result
    

    def get_candidates(self):
        

        # Getting iceberg data for positive concepts
        candidates_data = iceberg_operations.get_iceberg_data(
            self.concepts,
            self.goal_attr,
            len(self.objects_),
            self.baseline_counts,
        )

        return candidates_data

    def get_qubo_data(self, candidates, context, alpha, beta, n_rules) -> List[Dict]:
        Q_matrix, offset = qubo_formulation.build_qubo(candidates, context, alpha, beta, n_rules)
        sel_vec, energy = classical_solutions.solve_qubo_sim_anneal(Q_matrix, offset=offset, n_iters=2000, temp_start=1.0, temp_end=1e-3, seed=42)

        selected_candidates = [candidates[i] for i in range(len(candidates)) if sel_vec[i] == 1]
        return selected_candidates

    def classify_(self, undetermined_context: int) -> Dict[int, float] | int | None:


        #baseline candidates
        candidates_data = self.get_candidates()

        # Getting iceberg data for unknown concepts

        class_counts = self.get_class_counts(undetermined_context)
        unknown_candidates_data = iceberg_operations.get_iceberg_data(
            self.concepts,
            self.goal_attr,
            len(self.objects_),
            self.baseline_counts,
            class_counts
        )

        #mergin all candidates data
        candidates_data = {**candidates_data, **unknown_candidates_data}

        # generating both the negative and positive hypotheses
        self.train()


        #classification based on the data
        if len(candidates_data) == 0:
            return self.baseline_counts.most_common(1)[0][0] # type: ignore

        # Get QUBO data
        qubo_data = self.get_qubo_data(candidates_data, self.positive_context, alpha=0.5, beta=1.0, n_rules=3)

        if len(qubo_data) == 0:
            return self.baseline_counts.most_common(1)[0][0] # type: ignore

        votes = Counter()
        for q in qubo_data:
            for c in q['candidates']:
                votes[c] += 1

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
        return count_ones(result) / data_size
