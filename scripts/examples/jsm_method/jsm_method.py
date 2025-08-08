from typing import List, Set, Tuple
from numpy import testing
from fca.utils.bitset import BitSetOperations
from fca.utils.tools import count_ones

bset_operations = BitSetOperations()

class JSMMethodApplication:
    def __init__(
            self, object_attr_encoded: List[int],
            pos_context: List[int], 
            negative_context: List[int], 
            goal_attr: int, 
            objects_: List[int],
            attributes_: List[int]
            ):
        self.extent_intent_relation = object_attr_encoded
        self.postive_hypotheses = []
        self.negative_hypotheses = []
        self.goal_attr = goal_attr
        self.positive_context = pos_context
        self.negative_context = negative_context
        self.objects_ = objects_
        self.attributes_ = attributes_

    def train(self):
        self.postive_hypotheses = self.get_hypotheses(type=True) #type == True indicates positive hypotheses
        self.negative_hypotheses = self.get_hypotheses(type=False) #type == False indicates negative hypotheses

    def get_hypotheses(self, type: bool) -> List[int]:
        """
            Args:
                type(bool): a boolean value indicating whether to return positive or negative hypotheses
            Returns:
                a set containing all of possible hypotheses of the indicated type
        """
        candidate_hyp = []
        for i in range(len(self.positive_context)):
            if type:
                if self.check_valid_hypothesis(self.positive_context[i], type=True):
                    candidate_hyp.append(self.positive_context[i])
            else:
                if self.check_valid_hypothesis(self.negative_context[i], type=True):
                    candidate_hyp.append(self.negative_context[i])
        return candidate_hyp
    

    def check_valid_hypothesis(self, hypothesis: int, type: bool) -> bool:
        is_valid = True
        if type == True:
            for i in range(len(self.positive_context)):
                if not bset_operations.__subset_of__(hypothesis, self.positive_context[i]):
                    is_valid = False
                    break
        elif type == False:
            for i in range(len(self.negative_context)):
                if bset_operations.__subset_of__(hypothesis, self.negative_context[i]):
                    is_valid = False
                    break
        return is_valid

    def classify_(self, undetermined_context: List[int]) -> bool | str:
        has_negative = False
        has_positive = False


        #postive hypotheses test 
        for context_u in undetermined_context:
            for context_n in self.negative_context:
                if bset_operations.__subset_of__(context_u, context_n):
                    has_negative = True

        #postive hypotheses test
        for context_u in undetermined_context:
            for context_p in self.positive_context:
                if bset_operations.__subset_of__(context_u, context_p):
                    has_positive = True

            
        
        if has_positive and not has_negative:
            return True
        elif has_negative and not has_positive:
            return False
        elif has_positive and has_negative:
            return 'contradictory'
        else:
            return 'undetermined'


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
        result = (1 << data_size) - 1
        i = 0
        while data_size > 0:
            if (predictions & 1) != (testing_target & 1):
                result ^= (1 << i)
            i += 1
            data_size -= 1
        return result
