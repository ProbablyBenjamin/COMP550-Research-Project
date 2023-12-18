from gpt_label import _query_GPT, _get_seed_instances, _generate_labels, _query_GPT_json
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

def experiment_changing_return_format_json(seed_set_generation_method=random_split, seed_set_size=100, temperature=0, top_p=0.1, seed=1234):
    """Experimenting with constraining output file to json format"""
    
    #generate seed set
    seed_labels =_get_seed_instances(seed_set_size, seed_set_generation_method=seed_set_generation_method) #writes seed instances out to file
    #pass through gpt with default prompts
    _generate_labels(query_method=_query_GPT_json, temperature=temperature, top_p=top_p, seed=seed) #all default values

    accuracy = _evaluate_accuracy(seed_labels, convert_to_indices=False)

    print(f"Accuracy for json_enforced prompt for method {seed_set_generation_method}, with seedset size {seed_set_size}: {accuracy*100}%")

    open('COMP550-Research-Project/gpt_generated_labels.txt', 'w').close() #reset file

def experiment_hyperparameter_gridsearch(seed_set_size=10, top_ps=[0.1, 0.01, 0.001], temperatures=[0, 1, 2], seed_set_generation_methods=[random_split,uniform_split]):
    """Run grid search for best combination of hyperparams"""

    for i in range(0,len(top_ps)):
        for j in range(0, len(temperatures)):
            for k in range(0, len(seed_set_generation_methods)):

                seed_labels =_get_seed_instances(seed_set_size, seed_set_generation_method=seed_set_generation_methods[k])

                _generate_labels(query_method=_query_GPT_json, temperature=temperatures[i], top_p=top_ps[j])
                
                accuracy = _evaluate_accuracy(seed_labels, convert_to_indices=False)
                
                print(f"Accuracy for json enforced prompt for method {seed_set_generation_methods[k]}, with seedset size {seed_set_size}, top_p: {top_ps[i]}, temperature: {temperatures[j]}: {accuracy*100}%")
                
                open('COMP550-Research-Project/gpt_generated_labels.txt', 'w').close() #reset file

def experiment_preprocess_prompt(seed_set_generation_method=random_split, seed_set_size=100, temperature=0, top_p=1, seed=1234):
    """Experimenting with changing preprocessed vs. unpreprocessed prompt"""
    
    #generate seed set
    seed_labels =_get_seed_instances(seed_set_size, seed_set_generation_method=seed_set_generation_method) #writes seed instances out to file
    #pass through gpt with default prompts
    _generate_labels(temperature=temperature, top_p=top_p, seed=seed) #all default values

    accuracy = _evaluate_accuracy(seed_labels, convert_to_indices=True)

    print(f"Accuracy for default prompt for method {seed_set_generation_method}, with seedset size {seed_set_size}: {accuracy*100}%")

    modified_prompt = "For the following passage, tell me which domain the written text originates from out of the following possible domains:\
        Computer  Science, Electrical  Engineering,  Psychology,  Mechanical  Engineering, Civil  Engineering,  Medical  Science,  Biochemistry. \
        Do not return any text in your answer. Only return the index corresponding to the predicted domain. The indices range from 0 to 6. For example, a prediction of Computer Science\
        should return only the number 0"

    open('COMP550-Research-Project/gpt_generated_labels.txt', 'w').close() #reset file

    _generate_labels(base_prompt=modified_prompt, temperature=temperature, top_p=top_p, seed=1234) 

    accuracy = _evaluate_accuracy(seed_labels)

    print(f"Accuracy for modified prompt for method {seed_set_generation_method}, with seedset size {seed_set_size}: {accuracy*100}%")

    open('COMP550-Research-Project/gpt_generated_labels.txt', 'w').close() #reset file

def experiment_changing_prompt_detail(seed_set_generation_method=random_split, seed_set_size=100, temperature=0, top_p=0.1, seed=1234):
    """Experimenting with changing level of detail for prompt specification"""
    
    #generate seed set
    seed_labels =_get_seed_instances(seed_set_size, seed_set_generation_method=seed_set_generation_method) #writes seed instances out to file
    #pass through gpt with default prompts
    _generate_labels(temperature=temperature, top_p=top_p, seed=seed) #all default values

    accuracy = _evaluate_accuracy(seed_labels, convert_to_indices=True)

    print(f"Accuracy for default prompt for method {seed_set_generation_method}, with seedset size {seed_set_size}: {accuracy*100}%")

    modified_prompt = "For the following passage, tell me which domain the written text originates from out of the following possible domains:\
        Computer  Science, Electrical  Engineering,  Psychology,  Mechanical  Engineering, Civil  Engineering,  Medical  Science,  Biochemistry. \
        Do not return any text in your answer. Only return the index corresponding to the predicted domain. The indices range from 0 to 6. For example, a prediction of Computer Science\
        should return only the number 0"

    open('COMP550-Research-Project/gpt_generated_labels.txt', 'w').close() #reset file

    _generate_labels(base_prompt=modified_prompt, temperature=temperature, top_p=top_p, seed=1234) 

    accuracy = _evaluate_accuracy(seed_labels)

    print(f"Accuracy for modified prompt for method {seed_set_generation_method}, with seedset size {seed_set_size}: {accuracy*100}%")

    open('COMP550-Research-Project/gpt_generated_labels.txt', 'w').close() #reset file

if __name__ == "__main__":
    #experiment_changing_prompt_detail(seed_set_generation_method=random_split, seed_set_size=10)
    #experiment_changing_return_format_json(seed_set_generation_method=random_split, seed_set_size=10)
    #experiment_hyperparameter_gridsearch(seed_set_size=105) #multiple of 7 for easy distributioning
    print('hello')


