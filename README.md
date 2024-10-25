# Switch_Magic

This repository contains code and resources for the **Switch_Magic** project, which aims to explore the routing mechanism of Google Switch Transformer. Below is an overview of the files and directories included in this project and their purposes.



## File Overview

| File/Directory               | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| transformers_cp/             | a copy of the source code for Google Switch Transformer, made some changes in there to enable router history tracking |
| plots/                       | a place for saving the plots                                 |
| get_confidence_per_expert.py | script to get the attention confidence                       |
| plot_heat_map.py             | helper functions for plots                                   |
| inference_test.py            | a test file for inference                                    |
| inference_de_en.py           | a inference script for German to English translation         |
| get_de_en_LRP.py             | a script to get the layerwise relevance propagation          |
| train_de_en.py               | the training script to train a wmt de-en dataset             |
| `README.md`                  | This documentation file.                                     |