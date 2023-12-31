'''
https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/9rw3vkcfy4-6.zip

Download the dataset from this link -> extract -> extract WebOfScience.zip to get to the dataset files we want
Folder Structure should look similar to this:
dataset
    WOS
        Meta-data
        WOS5736
        ...
        ...
    get_dataset.py
    preprocessed_x.txt [contains preprocessed strings for the WOS11967 dataset]
'''

def get_instances(dataset = "WOS11967"):
    f = open(f'dataset/WOS/{dataset}/X.txt')
    x = f.read().splitlines()
    return x

def get_labels(dataset = "WOS11967"):
    f = open(f'dataset/WOS/{dataset}/YL1.txt')
    y = f.read().splitlines()
    return [int(x) for x in y]

def get_preprocessed_instances():
    f = open('dataset/preprocessed_x.txt')
    x = f.read().splitlines()
    return x

def get_llm_labels():
    f = open('dataset/gpt_generated_labels.txt')
    y = f.read().splitlines()
    return [int(x) for x in y]

if __name__ == "__main__":
    print(get_instances()[1])
    print(get_labels()[1])
