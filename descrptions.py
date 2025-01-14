from logging import raiseExceptions

import pandas as pd
import gensim.downloader as api
from nltk.corpus import wordnet as wn
import asyncio
import aiohttp
from googletrans import Translator
from sklearn.metrics.pairwise import cosine_similarity
import re
# Path to the Excel file
dataset_path = "./Final5_CatDataset.xlsx"
sheet_name = "Sheet1"
dataset_path2=r"C:\Users\matei\Downloads\RE7_modified_cat_database.xlsx"
sheet2_name="Code"
race1 = "Bengal"
race2="Ragdoll"
lang = "fra"
negligable_dif=0.5
# Load pre-trained Word2Vec model (Google News in this case)
word2vec_model = api.load("word2vec-google-news-300")


async def translate_to_english(french_word):

    translator = Translator()
    translation = await translator.translate(french_word, src='fr', dest='en')
    return translation.text


async def get_synonym(french_word):

    adjective_synsets = [synset for synset in wn.synsets(french_word, lang=lang) if synset.pos() == 'a']
    if adjective_synsets:
        lemma_list=list(adjective_synsets[0].lemmas())
        return lemma_list[0].name()
    else:
        print(f"WordNet didn't find synonyms for '{french_word}', using Google Translate.")
        return await translate_to_english(french_word)


async def get_antonym_wordnet(french_word):

    # Translate French to English
    english_word = await translate_to_english(french_word)
    synsets = wn.synsets(english_word)

    # Loop through synsets to find antonyms
    for synset in synsets:
        for lemma in synset.lemmas():
            if lemma.antonyms():
                return lemma.antonyms()[0].name()

    return None


async def get_antonym_word2vec(french_word):

    # Translate French to English
    english_word = await translate_to_english(french_word)

    try:
        similar_words = word2vec_model.most_similar(english_word, topn=10)
    except KeyError:
        print(f"Word '{english_word}' not in vocabulary.")
        return None

    antonyms = []
    word_vector = word2vec_model[english_word]

    # Compare each similar word's vector to the input word's vector using cosine similarity
    for similar_word, similarity in similar_words:
        similar_vector = word2vec_model[similar_word]
        cosine_distance = cosine_similarity([word_vector], [similar_vector])[0][0]  # Use cosine similarity here

        # If the cosine distance is high (i.e., similarity is low), consider it as a potential antonym
        if cosine_distance < 0.3:  # This threshold can be adjusted
            antonyms.append(similar_word)

    if antonyms:
        return antonyms[0]
    else:
        raise Exception("No antonym found in word2vec")



async def make_sentence_adjective(race, att):
    print(f"The cats from race {race} are very {att}")


# Preprocessing and Feature Extraction Functions
def get_useless_columns(df):
    all_columns = df.columns.to_list()
    word_to_search = ["Age", "age", "Zone", "Logement", "Row", "Nombre", "Sexe", "Race", "Abondance","Pred","Ext"]
    useless_columns = []
    for col in all_columns:
        useless = False
        for word in word_to_search:
            if word in col:
                useless = True
        if useless:
            useless_columns.append(col)
    return useless_columns



def extract_features_particular_race(averages):
    new_dict = {}
    for attribute in averages.keys():
        if averages[attribute] < 2 or averages[attribute] > 4:
            new_dict[attribute] = float(averages[attribute])
    return new_dict

async def get_race_abreviation(race):
    df = pd.read_excel(dataset_path2, sheet_name=sheet2_name)


    race_name_tokens=re.split(r'/', df.loc[2,"Meaning"])
    index=-1
    found_string=False
    for ind,token in enumerate (race_name_tokens):
        if race in token:
            index=ind
            found_string=True
            print("Found race in dataset")
            break

    if not found_string:
        raise Exception(f"{race} not classified in dataset")
    print(index)
    race_abrev_tokens=re.split(r'/', df.loc[2,"Values"])
    print(race_abrev_tokens)
    return race_abrev_tokens[index]


def get_main_features(race):
    df = pd.read_excel(dataset_path, sheet_name=sheet_name)
    filtered_rows = df[df["Race"] == race]
    # print(get_useless_columns(df))
    columns_to_average = filtered_rows.drop(columns=get_useless_columns(df))
    # print(columns_to_average)
    averages = dict(columns_to_average.mean())
    # print(averages)
    return extract_features_particular_race(averages)



async def get_description_race(race):
    try:
        race_abreviation=await get_race_abreviation(race)
        attributes_dict = get_main_features(race_abreviation)
        print("Attributes got")
        for att in attributes_dict.keys():
            try:
                if attributes_dict[att] > 4:
                    att = str(att)
                    print(f"Trying to find synonym for {att}...")
                    translated_word = await get_synonym(att)
                    print(f"Synonym for {att}: {translated_word}")
                else:
                    print(f"Trying to find antonym for {att}...")
                    translated_word = await get_antonym_wordnet(att)
                    if not translated_word:
                        print(f"WordNet didn't find antonym, trying Word2Vec for {att}...")
                        translated_word = await get_antonym_word2vec(att)
                    print(f"Antonym for {att}: {translated_word}")

                await make_sentence_adjective(race, translated_word)
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)
async def compare_races(race1,race2):
    race1_abreviation=await get_race_abreviation(race1)
    race2_abreviation=await get_race_abreviation(race2)


    att_race1_dict=get_main_features(race1_abreviation)
    att_race2_dict=get_main_features(race2_abreviation)
    att_race1_set=set(att_race1_dict.keys())
    att_race2_set=set( att_race2_dict.keys())


    common_att_set=att_race1_set & att_race2_set
    only_race1_att=att_race1_set-att_race2_set
    only_race2_att=att_race2_set-att_race1_set

    print(att_race1_dict)
    print(att_race2_dict)
    for att in common_att_set:
        quantifier1=att_race1_dict[att]
        quantifier2=att_race2_dict[att]
        dif=quantifier1-quantifier2
        english_word=await translate_to_english(att)
        if abs(dif)<negligable_dif:
            print(f"Race {race1} is equally {english_word} as race {race2}")
        else:
            if quantifier1>quantifier2:
                 print(f"Race {race1} is slightly more  {english_word} than race {race2}")
            else:
                 print(f"Race {race2} is slightly more  {english_word} than race {race1}")


    for att in only_race1_att:
       try:
            english_word=None
            if att_race1_dict[att]<2:

               english_word = await get_antonym_wordnet(att)
               if not english_word:
                    english_word = await get_antonym_word2vec(att)
            else:
                english_word=await get_synonym(att)
            if  english_word:
                      print(f"Race {race1} is more  {english_word} than race {race2}")
       except Exception as e:

            print(e)

    for att in only_race2_att:
       try:
            english_word=None
            if att_race2_dict[att]<2:

               english_word = await get_antonym_wordnet(att)
               if not english_word:
                    english_word = await get_antonym_word2vec(att)
            else:
                english_word=await get_synonym(att)
            if  english_word:
                      print(f"Race {race2} is more  {english_word} than race {race1}")
       except Exception as e:
            print(e)

# Call the function to see the results
async def main():
     # await get_description_race(race1)
     await compare_races(race1,race2)


# Running the async code
asyncio.run(main())
