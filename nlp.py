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

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

import torch
torch.zeros(1).cuda()



# +
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

# +
index_list = []
text_list = []
for book, page in q[:10]:
#     page_text = []
    for elem in page:
        if elem.tag == "text":
#             print(elem.attrib, elem.text)
            index_list.append({"book":book, "location":elem.attrib})
            text_list.append(elem.text)
#             page_text.append(elem.text)
#     if len(page_text) > 0:
#         text_list.append(" ".join(page_text))


# inputs    = tokenizer(sentence, return_tensors="pt").to(device)
# model     = model.to(device)
# outputs   = model(**inputs)


tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
model = BertModel.from_pretrained('skt/kobert-base-v1').to(device)
inputs = tokenizer.batch_encode_plus(text_list, padding=True, max_length=256,
                                    return_tensors="pt").to(device)
out = model(input_ids = torch.tensor(inputs['input_ids']),
              attention_mask = torch.tensor(inputs['attention_mask']))

result= list(zip(index_list, out.pooler_output.to("cpu").detach().numpy()))
# -








