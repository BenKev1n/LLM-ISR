# LLM-ISR: Interest-based Sequential Recommendation via Large Language Models for Long-Tail Users

This is the implementation of the paper "LLM-ISR: Interest-based Sequential Recommendation via Large Language Models for Long-Tail Users".

LLM-ISR is designed for long-tail sequential recommendation. It introduces LLM-based semantic embeddings into a sequential recommender, models long-term and short-term user interests separately, enhances short-term interests with cluster-aware semantic prototypes, and fuses the two interests with a dynamic gate.

To ease reproducibility, we archive the code and processed reproducibility files at Zenodo:

https://doi.org/10.5281/zenodo.20180506

## Configure the environment

To ease the configuration of the environment, I list versions of my hardware and software equipments:

- Hardware:
  - GPU: Quadro RTX 8000 48GB
  - Cuda: 12.4
  - Driver version: 550.144.03
  - CPU: Intel(R) Xeon(R) Gold 6234
- Software:
  - Python: 3.9.5
  - Pytorch: 2.6.0+cu124

You can conda install the environment.yml or pip install the requirements.txt to configure the environment.

    conda env create -f environment.yml

or

    pip install -r requirements.txt

## Dataset files

We conduct experiments on Yelp, Beauty, and Fashion.

1. The Yelp dataset can be obtained from https://www.yelp.com/dataset.
2. The Beauty and Fashion datasets are from the Amazon Review Data curated by McAuley Lab, available at https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/.

The raw dataset files downloaded from the websites should be put into:

    data/<yelp/fashion/beauty>/raw/

The original third-party datasets are not redistributed in this repository. Please download them from the official websites above if you want to run the preprocessing from raw data.

For direct reproduction, the processed files are provided in the Zenodo archive:

    https://doi.org/10.5281/zenodo.20180506

## Preprocess the dataset

You can preprocess the dataset and get the LLM embeddings according to the following steps.

1. Put the raw data into data/<yelp/fashion/beauty>/raw/.

2. Run `data/data_process.py` to filter cold-start users and items, sort each user interaction sequence by timestamp, re-map raw user and item IDs into consecutive integer IDs, and extract item metadata or category information.

   Run:

    python data/data_process.py

   Output files:

    data/<dataset>/handled/id_map.json
    data/<dataset>/handled/inter_seq.txt

   Note: the bottom of `data/data_process.py` currently contains hard-coded dataset calls. Switch the dataset call there before running it for `yelp`, `fashion`, or `beauty`.

3. Run `data/convert_inter.ipynb` to convert the sequence interaction file into the format used by this repository.

   Run:

    jupyter notebook data/convert_inter.ipynb

   Output file:

    data/<dataset>/handled/inter.txt

   The format of `inter.txt` is:

    user_id item_id

4. Run `data/<dataset>/get_item_embedding.ipynb` to build LLM item semantic embeddings.

   Run one of:

    jupyter notebook data/yelp/get_item_embedding.ipynb
    jupyter notebook data/fashion/get_item_embedding.ipynb
    jupyter notebook data/beauty/get_item_embedding.ipynb

   Output file:

    data/<dataset>/handled/itm_emb_np.pkl

5. Run `data/<dataset>/get_user_embedding.ipynb` to build LLM user semantic embeddings.

   Run one of:

    jupyter notebook data/yelp/get_user_embedding.ipynb
    jupyter notebook data/fashion/get_user_embedding.ipynb
    jupyter notebook data/beauty/get_user_embedding.ipynb

   Output file:

    data/<dataset>/handled/usr_emb_np.pkl

6. Run `data/pca.ipynb` to reduce the LLM item embedding dimension and obtain the collaborative-view initialization used by the model.

   Run:

    jupyter notebook data/pca.ipynb

   Output file:

    data/<dataset>/handled/pca64_itm_emb_np.pkl

7. Run `data/retrieval_users.ipynb` to build semantically similar users from the user LLM embeddings.

   Run:

    jupyter notebook data/retrieval_users.ipynb

   Output file:

    data/<dataset>/handled/sim_user_100.pkl

8. Run `data/K-means-user.ipynb` to build user cluster labels, soft cluster scores, and cluster centers for the Cluster MoE and prototype-based tail-user enhancement.

   Run:

    jupyter notebook data/K-means-user.ipynb

   Output files:

    data/<dataset>/handled/user_label.pkl
    data/<dataset>/handled/head_usr_emb_np.pkl
    data/<dataset>/handled/tail_usr_emb_np.pkl

9. Run `data/run_item_category_processing.py` to build item category labels for the auxiliary category prediction task.

   Run:

    python data/run_item_category_processing.py

   Output file:

    data/<dataset>/handled/item_label.pkl

10. Run `data/user_group.ipynb` if you want to explicitly generate head/tail user split files and related grouped-user artifacts.

   Run:

    jupyter notebook data/user_group.ipynb

   Output files:

    data/<dataset>/handled/head_user_id.txt
    data/<dataset>/handled/tail_user_id.txt
    data/<dataset>/handled/head_sim_user_100.pkl
    data/<dataset>/handled/tail_sim_user_100.pkl

In conclusion, the prerequisite files to run the main LLM-ISR experiments are as follows:

    inter.txt
    itm_emb_np.pkl
    usr_emb_np.pkl
    pca64_itm_emb_np.pkl
    sim_user_100.pkl
    user_label.pkl
    item_label.pkl

The current experiment scripts assume these files already exist under data/<dataset>/handled/. They do not call the raw-data preprocessing or LLM embedding generation steps online.

## Code information

The main files and folders are:

    main.py                         main entry for training, testing, and embedding export
    experiments/                    running scripts for Yelp, Fashion, and Beauty
    models/LLMISR.py                implementation of LLM-ISR
    models/DualLLMSRS.py            dual-view semantic/collaborative sequence encoder
    trainers/                       training, validation, testing, and metric reporting
    generators/                     dataset loading, sequence construction, and negative sampling
    utils/                          metrics, logging, early stopping, and helper functions
    data/data_process.py            raw-data preprocessing utilities
    analysis/                       scripts for result analysis and plotting

The default model in this repository is llmisr_sasrec.

## Run and test

You can reproduce the main experiments by running the bash scripts as follows:

    bash experiments/yelp.bash
    bash experiments/fashion.bash
    bash experiments/beauty.bash

Each script runs three random seeds by default: 42, 43, and 44.

The logs and results will be saved in the folder log/. The checkpoints will be saved in the folder saved/.

To test a saved checkpoint, use --do_test with the corresponding dataset and checkpoint path. For example:

    python main.py --dataset yelp --model_name llmisr_sasrec --check_path refined_seed42 --do_test

## Model pipeline

During training, the code follows this pipeline:

1. Load inter.txt and split each user sequence by leave-one-out evaluation. The last item is used for testing, the second-to-last item is used for validation, and the remaining items are used for training.
2. Construct sequence-to-sequence training samples with positive next-item labels and randomly sampled negative items.
3. Load LLM item embeddings from itm_emb_np.pkl and PCA item embeddings from pca64_itm_emb_np.pkl.
4. Encode the full sequence as long-term interest.
5. Keep the most recent ts_user interactions and encode them as short-term interest.
6. Enhance short-term interest with the Cluster MoE module using user_label.pkl.
7. Fuse long-term and short-term interests with a dynamic gate.
8. Optimize the recommendation loss together with user-alignment loss, tail-user prototype contrastive loss, and category prediction auxiliary loss.
9. Evaluate the model with HR@10 and NDCG@1/5/10/20, including short/long user groups and tail/popular item groups.

## Reproducibility notes

- The LLM semantic embeddings are precomputed offline and loaded from processed files.
- The experiment scripts pass --freeze, so the LLM item embedding table is frozen during training.
- Similar users, item labels, and user cluster files are offline preprocessing outputs.
- Results may have small numerical differences across different CUDA, GPU, and PyTorch versions.
- If the processed files are downloaded from Zenodo, you can run the experiment scripts directly without rerunning raw-data preprocessing.

## License and contribution guidelines

This repository is provided for academic research and reproducibility. Please check the license terms of Yelp and Amazon Review Data before downloading or using the datasets. Contributions and reproducibility reports are welcome.

## Thanks

The code refers to the repo LLM-ESR (https://github.com/liuqidong07/LLM-ESR/tree/master), SASRec (https://github.com/kang205/SASRec), and related sequential recommendation implementations.
