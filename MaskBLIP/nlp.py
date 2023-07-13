import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

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

# def get_noun_chunks(captions, spacy_model, include_background=False):
#     all_chunks = []
#     for cap in captions:
#         cap = remove_repeated_words(cap)
#         doc = spacy_model(cap)
#         chunks = [str(chunk) for chunk in doc.noun_chunks]
#         chunks = remove_articles(chunks)
#         all_chunks += chunks
#
#     #all_chunks = break_down_chunks(all_chunks, spacy_model)
#     #all_chunks = filter_substrings(all_chunks)
#     if include_background:
#         all_chunks += ['background']
#
#     all_chunks = list(set(all_chunks))
#     if not include_background and 'background' in all_chunks:
#         all_chunks.remove('background')
#     return all_chunks

def get_nouns(captions, lemmatizer):
    nouns = []
    for caption in captions:
        tokens = nltk.word_tokenize(caption)  # Tokenize the sentence
        tagged = nltk.pos_tag(tokens)  # Get the Part of Speech (POS) of each token
        nouns += [lemmatizer.lemmatize(word) for word, pos in tagged if pos in ['NN', 'NNS']]  # Extract the nouns and lemmatize them
    return list(set(nouns))

def map_labels(labels, class_list, class_embeds, model, threshold):
    mapped_labels, selected_idx, closest_values = [], [], []
    label_embeds = model.encode(labels, convert_to_tensor=True)
    for i, label_emb in enumerate(label_embeds):
        distances = torch.nn.CosineSimilarity(dim=1, eps=1e-6)(label_emb, class_embeds)
        closest_value = torch.amax(distances)
        if closest_value < threshold:
            continue
        selected_idx.append(i)
        closest_idx = torch.argmax(distances)
        mapped_labels.append(class_list[closest_idx])
        closest_values.append(closest_value.item())
    return mapped_labels, selected_idx, closest_values


if __name__ == "__main__":
    #nltk.download('punkt')
    #nltk.download('averaged_perceptron_tagger')
    #nltk.download('wordnet')
    text = ['a picture of owl owl three owls owl owl owl owl burr owl small', 'a picture of several small birds pose on short grass in the sun', 'a picture of this a mushrooms there a two leaf altered the trees a', 'a picture of a photograph of the ground from across the yard', 'a picture of an image of a group of four birds on the ground', 'a picture of owls owls owl owl horned three owl small owl owl owls', 'a picture of three baby birds sitting in the grass and staring up', 'a picture of altered a there this there the dark a a a we', 'a picture of this is an image of the grass in the sun', 'a picture of there are three smaller birds next to each other', 'a picture of owls four owl three owls owls owl owl five evil small', 'a picture of several small birds perched in grass and some green', 'a picture ofrum three the a a some a small a a lot', 'a picture of the grass has very little little vegetation on it', 'a picture of five birds on the grass in an open space']
    lemmatizer = WordNetLemmatizer()
    nouns = get_nouns(text, lemmatizer)
    print(nouns)

    # model = SentenceTransformer('all-MiniLM-L6-v2').cuda()
    #
    # # Tokenize input
    # CITYSCAPES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'trafffic sign',
    #               'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    #               'bicycle']
    # class_embeds = model.encode(CITYSCAPES, convert_to_tensor=True)
    #
    # labels = ['tree', 'trees', 'bushes', 'plants', 'street', 'bike', 'pedestrian', 'skyscraper', 'scooter', 'palm tree', 'woman', 'clouds', 'grass', 'stop sign', 'mountain', 'fence']
    # mapped_labels = map_labels(labels, CITYSCAPES, class_embeds, model)
    #
    # print(mapped_labels)

