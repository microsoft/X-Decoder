import spacy
import torch

from sentence_transformers import SentenceTransformer, util
#import torch
#import en_core_web_sm
def filter_substrings(noun_chunks):
    noun_chunks = list(set(noun_chunks))
    filtered_chunks = []
    for i, chunk in enumerate(noun_chunks):
        is_substring = False
        for j, other_chunk in enumerate(noun_chunks):
            if i != j and chunk in other_chunk.split():
                is_substring = True
                break
        if not is_substring and not chunk in ['a', 'an', 'the', "the a"]:
            filtered_chunks.append(chunk)
    return filtered_chunks

def remove_articles(noun_chunks):
    clean_chunks = []
    for chunk in noun_chunks:
        if chunk.lower().startswith('the '):
            clean_chunks.append(chunk[4:])
        elif chunk.lower().startswith('a '):
            clean_chunks.append(chunk[2:])
        elif chunk.lower().startswith('an '):
            clean_chunks.append(chunk[3:])
        else:
            clean_chunks.append(chunk)
    return clean_chunks

def remove_repeated_words(cap):
    chunk_text = cap.split()
    chunk_text_no_repeats = [word for i, word in enumerate(chunk_text) if (word != chunk_text[i-1] and i>0) or i==0]
    chunk = ' '.join(chunk_text_no_repeats)
    return chunk
def load_spacy():
    nlp = spacy.load('en_core_web_sm')
    #nlp = en_core_web_sm.load()
    return nlp
def get_noun_chunks(captions, spacy_model, include_background=False):
    all_chunks = []
    for cap in captions:
        cap = remove_repeated_words(cap)
        doc = spacy_model(cap)
        chunks = [str(chunk) for chunk in doc.noun_chunks]
        chunks = remove_articles(chunks)
        all_chunks += chunks

    #all_chunks = break_down_chunks(all_chunks, spacy_model)
    #all_chunks = filter_substrings(all_chunks)
    if include_background:
        all_chunks += ['background']

    all_chunks = list(set(all_chunks))
    if not include_background and 'background' in all_chunks:
        all_chunks.remove('background')
    return all_chunks

def break_down_chunks(chunks, nlp):
    new_list = []
    for string in chunks:
        words = string.split()
        # check if a string has more than 3 words
        if len(words) > 3:
            # process the string with spacy
            doc = nlp(string)
            # check which words are nouns and add them to the new list
            for token in doc:
                if token.pos_ == 'NOUN':
                    new_list.append(token.text)
        else:
            new_list.append(string)
    return new_list

def get_nouns(captions, spacy_model, add_background=False):
    all_nouns = []
    for cap in captions:
        cap = remove_repeated_words(cap)
        doc = spacy_model(cap)
        nouns = [token.lemma_ for token in doc if token.pos_ == 'NOUN']
        if add_background:
            nouns += ['background']
        all_nouns += nouns

    output = list(set(all_nouns))
    if not add_background and 'background' in output:
        output.remove('background')
    return output

def map_labels(labels, class_list, class_embeds, model):
    label_embeds = model.encode(labels, convert_to_tensor=True)

    label_embeds = label_embeds.unsqueeze(1)  # shape will be [n, 1, 384]
    class_embeds = class_embeds.unsqueeze(0)  # shape will be [1, m, 384]

    similarity = torch.nn.functional.cosine_similarity(label_embeds, class_embeds, dim=2)  # shape will be [n, m]

    max_sim_values, max_sim_indices = torch.max(similarity, dim=1)

    # Apply the threshold
    mask = max_sim_values > 0.5
    filtered_sim_values = max_sim_values[mask]
    filtered_sim_indices = max_sim_indices[mask]

    # Convert to list
    filtered_sim_values_list = filtered_sim_values.tolist()
    filtered_sim_labels_list = [class_list[i] for i in filtered_sim_indices]

    return filtered_sim_labels_list, filtered_sim_values_list

if __name__ == "__main__":

    model = SentenceTransformer('all-MiniLM-L6-v2').cuda()

    # Tokenize input
    CITYSCAPES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'trafffic sign',
                  'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                  'bicycle']
    class_embeds = model.encode(CITYSCAPES, convert_to_tensor=True)

    labels = ['tree', 'trees', 'bushes', 'plants', 'street', 'bike', 'pedestrian', 'skyscraper', 'scooter', 'palm tree', 'woman', 'clouds', 'grass', 'stop sign', 'mountain', 'fence']
    mapped_labels = map_labels(labels, CITYSCAPES, class_embeds, model)

    print(mapped_labels)