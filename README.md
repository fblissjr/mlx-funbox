# mlx-funbox
mlx and mlx-lm CLI toolbox for my own personal use, and maybe yours too. primary built to test out [cohere's command-r plus model](https://docs.cohere.com/docs/command-r-plus) using mlx + cohere's open weights release on [hugging face](https://huggingface.co/CohereForAI/c4ai-command-r-plus). quantized weights are available on the mlx-community org on hugging face [here](https://huggingface.co/mlx-community/c4ai-command-r-plus-4bit)

## why is the cli app called fff.py?
because when i was creating it, i typed ```nano f``` and the key repeat rate was set too high, and i decided ¯\_(ツ)_/¯ 

## how do you use this?

pipe in whatever you like to cohere's model (or something else). this was primarily built 

example usage: ```cat fun.txt | python fff.py "Summarize" --temp 0.0 --model ./mlx-community_c4ai-command-r-plus-4bit```

example usage 2, using pbpaste (paste clipboard contents): ```pbpaste | python fff.py "Summarize" --temp 0.0 --model mlx-community_c4ai-command-r-plus-4bit --use-tools```

tools are currently just hardcoded per cohere's model card (https://huggingface.co/CohereForAI/c4ai-command-r-plus) but should absolutely be changed to be dynamic so you can use any tool. likely no time to do that today, but feel free to PR or clone or fork or whatever you like.
