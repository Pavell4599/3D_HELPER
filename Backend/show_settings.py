import json
from Ai.classyfier import predict_only       


#ПЕРЕМЕННЫЕ
guessed_class_name = 'vase'
TEST_PATH = r"C:\Users\Павел\Downloads\_Pugg.stl"
TEST_CLASS = 0


with open(f"Data/settings/{guessed_class_name}.json", "r", encoding="utf-8") as f:
    data = json.load(f)
for k, l in data.items():
    print(f'{k} ---> {l}')



print(next(iter(predict_only(TEST_PATH))))






