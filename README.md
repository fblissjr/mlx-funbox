# mlx-funbox
mlx and mlx-lm CLI toolbox for my own personal use, and maybe yours too

pipe in whatever you like to cohere's model (or something else). haven't tested tool use with the updated tokenizer fix.

example usage: ```cat fun.txt | python fff.py "Summarize" --temp 0.0 --model ./mlx-community_c4ai-command-r-plus-4bit```

example usage 2, using pbpaste (paste clipboard contents): ```pbpaste | python fff.py "Summarize" --temp 0.0 --model mlx-community_c4ai-command-r-plus-4bit --use-tools```

tools are currently just hardcoded per cohere's model card (https://huggingface.co/CohereForAI/c4ai-command-r-plus) but should absolutely be changed to be dynamic so you can use any tool. likely no time to do that today, but feel free to PR or clone or fork or whatever you like.
