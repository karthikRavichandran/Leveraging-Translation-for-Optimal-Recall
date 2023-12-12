#!pip install rank_bm25
from rank_bm25 import BM25Okapi, BM25L, BM25Plus 
import pandas as pd


#reading translated data
translated_df = pd.read_csv('chunk_0_20_en_es_en.csv').drop('Unnamed: 0', axis=1)

def find_non_intersection(row):
    set1 = set(row['original text'].lower().split())
    set2 = set(row['rev_translated'].lower().split())
    return set2 - set1.intersection(set2)

def find_intersection(row):
    set1 = set(row['original text'].lower().split())
    set2 = set(row['rev_translated'].lower().split())
    return set1.intersection(set2)

def combine_words(row):
    words_set = set(row['original text'].split()) | set(row['rev_translated'].split())
    return list(words_set)

# Apply the function to create a new column 'Intersection'
translated_df['non_intersection'] = translated_df.apply(find_non_intersection, axis=1)
translated_df['intersection'] = translated_df.apply(find_intersection, axis=1)


# Apply the function to create a new column 'CombinedWords'
translated_df['CombinedWords'] = translated_df.apply(combine_words, axis=1)
translated_df.to_csv('qry_expansion.csv')


#----------------------------------------

qry_expanded = translated_df['CombinedWords'].to_list()
just_qry = translated_df['original text'].to_list()
rev_qry = translated_df['rev_translated'].to_list()

def get_bm25_score(qry, profile, bm25_algo_given, bm25_th= 1.5, in_list=True, top_k = 5):
    corpus = []
    for j in profile:
        corpus.append(j['text'])
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    # bm25 = BM25Okapi(tokenized_corpus)
    bm25 = bm25_algo_given(tokenized_corpus)
    if not in_list:
        tokenized_query = qry.split(" ")
    else:
        tokenized_query = qry
    doc_scores = bm25.get_scores(tokenized_query)
    # print(doc_scores)
    # print(np.sum(doc_scores>bm25_th))
    return {"doc_scores": doc_scores, "extracted_doc":np.sum(doc_scores>bm25_th),
            "doc_passed": doc_scores> bm25_th,
            "topk_idx": np.argpartition(doc_scores, -top_k)[-top_k:]
           } 
def gen_prompt(qry, selected_profile_data): #just_qry[0]
    expm = [f'"{i}"' for i in selected_profile_data]
    return f'''rephrase the this phrase \"{qry}\" using user profile examples : [{" , ".join(expm)}]. Just give me the rephared sentence'''

just_qry_prompt = []
rev_qry_prompt = []
just_pluse_rev_expanded_prompt = []
for i in range(100):
    rev_qry_out = get_bm25_score(rev_qry[i], json_data[i]['profile'], BM25Plus, in_list=False, top_k = 2)
    just_qry_out = get_bm25_score(just_qry[i], json_data[i]['profile'], BM25Plus, in_list=False, top_k = 2)
    qry_expanded_out = get_bm25_score(qry_expanded[i], json_data[i]['profile'], BM25Plus, in_list=True, top_k = 2)

    selected_profile_data_just_qry = []
    selected_profile_data_rev_qry = []
    selected_profile_data_qry_expanded = []
    for idx, j in enumerate(json_data[i]['profile']):
        if idx in just_qry_out['topk_idx']:
            selected_profile_data_just_qry.append(j['text'])
        if idx in rev_qry_out['topk_idx']:
            selected_profile_data_rev_qry.append(j['text'])
        if idx in qry_expanded_out['topk_idx']:
            selected_profile_data_qry_expanded.append(j['text'])
    just_qry_prompt.append(gen_prompt(just_qry[i], selected_profile_data_just_qry))
    rev_qry_prompt.append(gen_prompt(just_qry[i], selected_profile_data_rev_qry))
    just_pluse_rev_expanded_prompt.append(gen_prompt(just_qry[i], selected_profile_data_just_qry + selected_profile_data_rev_qry))

final_prompts = {"just_qry_prompt" : just_qry_prompt, "rev_qry_prompt": rev_qry_prompt, 
                 "just_pluse_rev_expanded_prompt": just_pluse_rev_expanded_prompt}

file_path = "input_for_LLM.json"
# Save the dictionary to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(final_prompts, json_file)




