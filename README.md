Design, train, and perform inference with diffusion models
This is the code used for my undergraduate honors thesis entitled _Diffusion Models to Alleviate Class Imbalance_ during my senior year at Arizona State University
and the article _Generative Modeling with Diffusion_ found in the journal [SIURO](https://www.siam.org/publications/siam-journals/siam-undergraduate-research-online-siuro/) (arXiv link [here](https://arxiv.org/abs/2412.10948)).
Train diffusion models through main_network.py, and see UMAP_and_Classification.ipynb for a walkthrough of the creditcard dataset and comparing classifiers with different modes of data augmentation.

Examples of reverse diffusion process:

On the make_moons dataset from Scikit-learn:
![Alt text](https://github.com/justinle4/Diffusion/blob/main/reverse_diffusion_examples/scatterplot_timeline_moons.png)

On the Datasaurus dataset from datasets/datasaurus.csv:
![Alt text](https://github.com/justinle4/Diffusion/blob/main/reverse_diffusion_examples/scatterplot_timeline_datasaurus.png)
