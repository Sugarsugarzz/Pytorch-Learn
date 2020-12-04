"""

搜索引擎
    - 倒排索引（返回的结果依然数据量很大）
    - 关键词与文档内容相似度匹配算法
        - TF-IDF
        词频（TF)、逆文本频率指数（IDF）
        统计每篇文章词的词频、找出有区分力的词（出现次数多，区分力小）
        由候选文章的TF-IDF和搜索问句的TF-IDF进行cos相似度计算，
"""