You will find following Dir:
a) Tweet_Script_files	
b) news_Script_files	
c) other_files

In Tweet_Script_files and news_Script_files, you will find the translated Query in a CSV file,  input for LLM and LLM output. Along with these, you will find the Jupyter Notebook and 
python files to generate the above files.
```bash
├── Tweet_Script_files
│   ├── LLM_output_just_BM25_tweet_first_100_training.json : [LLM Output for just using BM25]
│   ├── LLM_output_just_pluse_rev_tweet_first_100_training.json : [LLM Output for translation pipeline using BM25]
│   ├── bm25_and_prompt_gen.py [BM25 Retrieval on normal and Translated query, and adding those to create a prompt for LLM  ]
│   ├── chunk_0_20_en_es_en.csv [20*50  = 1000 first query translation recorded]
│   ├── translation_colab.ipynb [Query Expansion using Translation using GPU ]
│   └── tweeter_translation_process.py [Query Expansion using Translation ]
├── news_Script_files
│   ├── LLM_output_just_news_first_100_training.json [LLM Output for just using BM25]
│   ├── LLM_output_just_pluse_rev_news_first_100_training.json [LLM Output for translation pipeline using BM25]
│   ├── fwd_rev_translation_news.py [Query Expansion using Translation ]
│   ├── input_for_LLM_news.json [LLM input json for news]
│   ├── news_150_en_es_en.csv [first 150  query translation recorded]
│   ├── news_LLM.py [LLM code for both Tweet and news]
│   └── result_generation_news.ipynb [Rough 1 and L for both new and tweet]
└── other_files [Other junk files for future reference]
    ├── bm25_and_prompt_gen.py
    ├── chunk_0_20_en_es_en.csv
    ├── chunk_10437_en_es_en.csv
    ├── fwd_rev_translation_news.py
    ├── news_translation.py
    ├── train_questions_tweet.json
    ├── translation_colab.ipynb
    └── tweeter_translation_process.py
```

    
