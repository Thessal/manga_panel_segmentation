# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# # !pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
# # !pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
# -

import xml.etree.ElementTree
et = xml.etree.ElementTree
import torch
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
torch.zeros(1).cuda()


def get_data():
    class temp:
        pass
    self = temp()

    manga109_path = "Manga109s_released_2021_12_30"
    manga109_ko_path = "Manga109_ko" # korean translated manga109 annotation

    with open(f"{manga109_path}/books.txt", "r") as f:
        self.books = [x.strip() for x in f.readlines()]

    q = []
    for book in self.books:
        with open(f"{manga109_ko_path}/ko_{book}.xml", 'r') as f:
            annotation = et.fromstring(f.read())
            pages = annotation[1]
            for page in pages:
                q.append((book, page))
    
    index_list = []
    text_list = []
    # for book, page in q[:10]:
    for book, page in q:
        for elem in page:
            if elem.tag == "text":
                index_list.append({"book":book, "location":elem.attrib})
                text_list.append(elem.text)
                
    return zip(index_list, text_list)


# tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
# model = BertModel.from_pretrained('skt/kobert-base-v1').to(device)
def embed(data, model, tokenizer):
    index_list, text_list = zip(*data)
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated(),torch.cuda.memory_reserved())

    if len(text_list) == 0:
        return []
    inputs = tokenizer.batch_encode_plus(text_list, padding=True, max_length=256,
                                        return_tensors="pt").to(device)
    out = model(
        input_ids = inputs['input_ids'],
        attention_mask = inputs['attention_mask'])
    
    results= list(zip(index_list, out.pooler_output.to("cpu").detach().numpy()))
    
    del inputs, out
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated(),torch.cuda.memory_reserved())
    print("==")
    
    return results


# +
import base64
import json
def serialize(results):
    result_serialized=[]
    for result in results:
        x = result[0].copy()
        x["text"] = base64.b64encode(result[1].astype(np.float32).tobytes()).decode('ascii')
        result_serialized.append(x)
    # s = json.dumps(result_serialized)
    return result_serialized

# with open("text_embedded.json", "w") as f:
#     json.dump(serialize(results),f)


# +
def deserialize(json):
    result_load = []
    for result in result_serialized:
        x = result.copy()
        x["text"] = np.frombuffer(base64.b64decode(result['text']), dtype=np.float32)
        result_load.append(x)
    return result_load

# # result_serialized = json.loads(s)
# with open("text_embedded.json", "r") as f:    
#     result_loaded = deserialize(json.load(f))


# -

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
model = BertModel.from_pretrained('skt/kobert-base-v1').to(device)

dataset = list(get_data())
batch_size = 128
for idx, i in enumerate(range(0,len(dataset),batch_size)):
    print(idx, i, i+batch_size, len(dataset))
    vectors = embed(dataset[i:i+batch_size], model, tokenizer)
    with open(f"./text_embedded/text_embedded_{idx}.json", "w") as f:
        json.dump(serialize(vectors),f)




