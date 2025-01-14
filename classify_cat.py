import numpy as np
import pandas as pd
import re
from nltk.corpus import wordnet
import nltk
import nn
import joblib
from nn import NeuralNetwork
from nn import Layer
from nn import race_mapping

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

keyword_mapping = {
    'behavior': {
        1: ['very low', 'absent', 'not at all', 'hardly', 'barely', 'none'],
        2: ['low', 'slightly', 'somewhat', 'a little', 'mildly','limited'],
        3: ['moderate', 'average', 'medium', 'somewhat'],
        4: ['high', 'very', 'quite', 'considerably', 'strongly','long'],
        5: ['very high', 'extremely', 'intensely', 'highly', 'completely', 'totally', 'utterly','all the time']
    },
    'predator': ['predator', 'hunter', 'carnivore', 'raptor', 'bird of prey', 'falcon', 'eagle', 'hawk', 'owl', 'lion', 'tiger', 'wolf', 'fox'],
    'age': ['age', 'year', 'years', 'old', 'baby', 'infant', 'newborn', 'adult', 'senior', 'juvenile'],
    'abundance': ['abundance', 'rare', 'common', 'plentiful', 'low', 'medium', 'high'],
    'housing': ['aviary', 'artificial', 'breeding'],
    'zone': ['urban', 'rural', 'protected', 'park'],
    'nombre': ['number','amount','quantity','cats','cat','household'],
    'ext': ['outside','outdoors','time','hour','hours'],
    'obs': ['time','observation','sighting','encounter','report','game','petting','care'],
    'sexe':['male','female','boy','girl']
}

def extract_keywords(description):
    words = nltk.word_tokenize(description.lower())
    tagged_words = nltk.pos_tag(words)
    print(tagged_words)
    keywords = [word for word, tag in tagged_words if tag.startswith('NN') or tag.startswith('JJ')]
    return keywords

def map_keywords(keywords, column_name):
    if column_name in ['PredOiseau', 'PredMamm', 'Logement_AAB', 'Logement_ASB', 'Logement_MIL', 'Logement_ML', 'Zone_PU', 'Zone_R', 'Zone_U', 'Abondance_1', 'Abondance_2', 'Abondance_3']: #Boolean attributes
        relevant_keywords = []
        for key, value in keyword_mapping.items(): #Find the correct keywords list
            if column_name.startswith(key.capitalize()): #check if the column name starts with the key
                relevant_keywords = value
                break
        if any(keyword in relevant_keywords for keyword in keywords):
            return True
        return False
    elif column_name in ['Timide','Calme','Effrayé','Intelligent','Vigilant','Perséverant','Affectueux','Amical','Solitaire','Brutal','Dominant','Agressif','Impulsif','Prévisible','Distrait','Ext','Obs']: #Scale attributes
        for scale, scale_keywords in keyword_mapping['behavior'].items():
            if any(keyword in scale_keywords or any(synonym in scale_keywords for synonym in get_synonyms(keyword)) for keyword in keywords):
                return scale
        if column_name == 'Ext' and any(keyword in keyword_mapping['ext'] for keyword in keywords):
            return 1 #Default if only ext is mentioned
        if column_name == 'Obs' and any(keyword in keyword_mapping['obs'] for keyword in keywords):
            return 1 #Default if only obs is mentioned
        return 'other' #Return other if no scale and no keyword for ext and obs

    elif column_name in ['Age_1a2', 'Age_2a10', 'Age_Moinsde1', 'Age_Plusde10']: #Age attributes
        if any(re.search(r'\b(1|2|10)\b', keyword) for keyword in keywords) or any(re.search(r'\b(1-2|2-10|less than 1|under 1|over 10|10\+)\b', keyword) for keyword in keywords):
            return 'age'

    elif column_name == 'Nombre': #Number attribute
        if any(keyword in keyword_mapping['nombre'] for keyword in keywords):
            for keyword in keywords:
                if keyword.isdigit():
                    return int(keyword)
            return 1

    elif column_name == 'Sexe': #Sexe attribute
        if any(keyword in keyword_mapping['sexe'] for keyword in keywords):
            return keywords[keywords.index([keyword for keyword in keywords if keyword in keyword_mapping['sexe']][0])] #return the actual keyword found

    return 'other'

def process_description(description, df_columns):
    keywords = extract_keywords(description)
    results = {}
    print("Extracted Keywords:", keywords)
    for col in df_columns:
        results[col] = map_keywords(keywords, col)
    return results

# Example usage
description = "A delicate and cautious cat with a plush, snow-white coat and striking blue eyes that seem to glimmer with intelligence. The cat spends most of its time exploring the quiet corners of the house or curling up in warm spots near the heater. It lives in a small town with a few gardens and open spaces nearby, occasionally watching birds from the safety of a window but never venturing out to chase them. This cat is female."

df_columns = ['Nombre','Ext','Obs','Timide','Calme','Effrayé','Intelligent','Vigilant','Perséverant','Affectueux','Amical','Solitaire','Brutal','Dominant','Agressif','Impulsif','Prévisible','Distrait','PredOiseau', 'PredMamm', 'Age_1a2', 'Age_2a10', 'Age_Moinsde1', 'Age_Plusde10', 'Logement_AAB', 'Logement_ASB', 'Logement_MIL', 'Logement_ML', 'Zone_PU', 'Zone_R', 'Zone_U', 'Abondance_1', 'Abondance_2', 'Abondance_3','Sexe']
print(len(df_columns))
mapping_results = process_description(description, df_columns)
print(mapping_results)

new_row = {}
for col, category in mapping_results.items():
    new_row[col] = category
tbd = np.array([100])
for k, v in new_row.items():
    if isinstance(v, int):
        tbd = np.append(tbd, v)
    elif v == 'other':
        tbd = np.append(tbd, np.nan)
    elif v == 'male' or v:
        tbd = np.append(tbd, 0)
    elif v == 'female' or not v:
        tbd = np.append(tbd, 1)
print(tbd)

NN = joblib.load('trained_data.pkl')
race_mapping = nn.race_mapping
nn.extract_data()
print(NN.classify(tbd))

