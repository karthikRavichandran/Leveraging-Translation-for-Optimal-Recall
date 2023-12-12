from transformers import MarianMTModel, MarianTokenizer
import torch
import json
class TranslateMain:
    def __init__(self, model_name_fwd='Helsinki-NLP/opus-mt-en-es', model_name_reverse='Helsinki-NLP/opus-mt-es-en'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_fwd = MarianMTModel.from_pretrained(model_name_fwd)
        self.tokenizer_fwd = MarianTokenizer.from_pretrained(model_name_fwd)
        self.model_reverse = MarianMTModel.from_pretrained(model_name_reverse)
        self.tokenizer_reverse = MarianTokenizer.from_pretrained(model_name_reverse)

    def translate_pass(self, texts, pass_fwd_or_rev):
        if pass_fwd_or_rev == 'fwd':
            tokenizer = self.tokenizer_fwd
            model = self.model_fwd.to(self.device)
        else:
            tokenizer = self.tokenizer_reverse
            model = self.model_reverse.to(self.device)
        # Tokenize input texts
        input_ids = tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True)['input_ids']
        
        input_ids = input_ids.to(self.device)
        # Perform translation
        output_ids = model.generate(input_ids)

        # output_ids = output_ids.to('cpu')

        # Decode the translated texts
        translated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        input_ids.to('cpu')
        return translated_texts

    def rephrase_using_translation(self, texts):
        fwd_pass_text = self.translate_pass(texts, pass_fwd_or_rev='fwd')
        print(f'Intermediate translation: {fwd_pass_text}')
        rev_pass_text = self.translate_pass(fwd_pass_text, pass_fwd_or_rev='rev')
        return fwd_pass_text, rev_pass_text

def get_remaining_string(original_string, substring):
    index = original_string.find(substring)

    if index != -1:
        # If the substring is found, return the part of the string after the substring
        remaining_string = original_string[index + len(substring):]
        return remaining_string
    else:
        # If the substring is not found, return the original string
        return original_string



# Opening JSON file
f = open('/content/train_questions.json')
data = json.load(f)

substring = "Paraphrase the following tweet without any explanation before or after it:"
only_the_tweet = []
for i in data:
    only_the_tweet.append(get_remaining_string(i['input'], substring))

from tqdm import tqdm
import pandas as pd
# len(original_text)
def split_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


original_list = only_the_tweet  
chunk_size = 50

result = split_list(original_list, chunk_size)
checkpoint = -1
# Print the result
list_of_df = []
translator = TranslateMain()
for i, chunk in tqdm(enumerate(result), total=len(result), desc="Processing transaltion"):
    if i>checkpoint:
      torch.cuda.empty_cache()
      original_text = chunk
      rephrased_texts = translator.rephrase_using_translation(original_text)
      df = pd.DataFrame({"original text": original_text, 'fwd_translated': rephrased_texts[0], 'rev_translated': rephrased_texts[1]})
      torch.cuda.empty_cache()
      list_of_df.append(df)

result.to_csv("chunk_10437_en_es_en.csv")


#Output:
'''
	original text	fwd_translated	rev_translated
0	I'm currently enjoying the album "Listen to E...	Actualmente estoy disfrutando del álbum "Escuc...	I am currently enjoying the album "Listen to E...
1	Visit the blog for the best quote of the day,...	Visite el blog para obtener la mejor cita del ...	Visit the blog to get the best date of the day...
2	@jackfaulkner suggested using heroin to help ...	@jackfaulkner sugirió usar heroína para ayudar...	@jackfaulkner suggested using heroin to help r...

'''
