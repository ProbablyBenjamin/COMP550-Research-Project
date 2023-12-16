from gpt_label import _query_GPT, _get_seed_instances, _generate_labels
from seed_sampler import random_split, uniform_split
import numpy as np

def _evaluate_accuracy(ground_truth, gpt_labels='COMP550-Research-Project/gpt_generated_labels.txt', convert_to_indices=False):
    """Return the accuracy of the gpt_labels as compared to actual ground-truth for a particular seed set"""

    with open(gpt_labels, 'r') as labels:
        predictions = labels.read().splitlines()

        if convert_to_indices:
            predictions = _convert_to_indices(predictions)
    
    return sum([1 for i in range(len(predictions)) if predictions[i] == ground_truth[i]])/len(predictions)

def _convert_to_indices(gpt_labels):
    
    categories = ["Computer Science", "Electrical Engineering",  "Psychology",  
                  "Mechanical Engineering", "Civil Engineering",  "Medical Science",  "Biochemistry"]

    converted_labels = [] 
    for label in gpt_labels:
        index_found=False
        for category in categories:
            if category in label:
                index = categories.index(category)
                index_found=True
                break
        if index_found:
            converted_labels.append(str(index))
        else:
            converted_labels.append('-1')

    return converted_labels

def experiment_changing_prompt_detail(seed_set_generation_method=random_split, seed_set_size=100):
    """Experimenting with changing level of detail for prompt specification"""
    
    #generate seed set
    seed_labels =_get_seed_instances(seed_set_size, seed_set_generation_method=seed_set_generation_method) #writes seed instances out to file
    #pass through gpt with default prompts
    _generate_labels() #all default values

    accuracy = _evaluate_accuracy(seed_labels, convert_to_indices=True)

    print(f"Accuracy for default prompt for method {seed_set_generation_method}, with seedset size {seed_set_size}: {accuracy*100}%")


if __name__ == "__main__":
    experiment_changing_prompt_detail(seed_set_generation_method=random_split, seed_set_size=10)

