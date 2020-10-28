import pickle



f = pickle.load(open(".\\Weibo_Data\\origin_data\\trump-content.pkl", "rb"))
file = open(".\\Weibo_Data\\origin_data\\trump_content.txt", "w", encoding="utf-8")
for i in f:
    file.write(i + "\n")