import numpy as np
import torch
import re

# with open("test.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
# print("Total number of character:", len(raw_text))
# print(raw_text[:99])

# text = "hi this is my\\ first ; ,time using regex in python ,, '' '' ' \"\" "
text1 = "Hello, world. This, is a test."
# vals = re.split(r'\s',text)
vals1 = re.split(r'([,.]|\s)',text1)
vals1 = [item for item in vals1 if item.strip()]
# print(len(vals))
# print(vals)
# print(text)
print(len(vals1))
print(vals1)