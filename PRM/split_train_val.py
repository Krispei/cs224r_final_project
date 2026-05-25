import modal

app = modal.App("Humble-shepherd-sft")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python='3.11'
    )
    .run_commands(
        "pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install([
         "transformers==4.46.0", "datasets", "accelerate", "wandb", "trl", "numpy", "vllm==0.6.3", "matplotlib"
    ])
)

volume = modal.Volume.from_name("Humble-shepherd-data")

@app.function(image=image, volumes={"/data": volume}, timeout=60*60)
def split_prm_dataset():
    '''
    splits annotated prm data into train and validation splits. Each annotation is a dictionary with keys:
    - 'prompt'
    - 'ground_truth'
    - 'label' 
    '''
    from datasets import load_from_disk, Dataset
    import random

    annotations_ds = load_from_disk('/data/prm_annotations')
    
    '''
    questions is a dict, with each key being a question, and each key stores a list of annotations
    '''
    questions = {}

    for annotation in annotations_ds:

        prompt = annotation['prompt']
        question = prompt.split('<|im_start|>user\n')[1].split('<|im_end|>')[0]

        if question not in questions:
            questions[question] = []

        questions[question].append(annotation)

    questions_list = list(questions.keys())

    random.shuffle(questions_list)

    split_indx = int(0.8 * len(questions_list))

    train_questions = questions_list[:split_indx] 
    val_questions = questions_list[split_indx:]

    train_annotations = []
    val_annotations = []

    for question in train_questions:
        for ann in questions[question]:
            train_annotations.append(ann)

    for question in val_questions:
        for ann in questions[question]:
            val_annotations.append(ann)

    print(f" Num Train annotations : {len(train_annotations)}")
    print(f" Num Val annotations : {len(val_annotations)}")

    train_ds = Dataset.from_list(train_annotations)
    val_ds = Dataset.from_list(val_annotations)

    train_ds.save_to_disk('/data/prm_train')
    val_ds.save_to_disk('/data/prm_val')

    volume.commit()

    print("Saved train and validation datasets")

        
@app.local_entrypoint()
def main():
    split_prm_dataset.remote()
                

