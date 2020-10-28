存储格式

[created_at, content原文, [content分词], keyword, weibo_url, like_num, repost_num, comment_num, tool]



<keyword>.pkl : 全部数据

<keyword>-content.pkl : 正文数据，仅包含正文的列表



文件内容获取

```python
import pickle

with open('trump.pkl', 'rb') as f:
    content = pickle.load(f)
```

