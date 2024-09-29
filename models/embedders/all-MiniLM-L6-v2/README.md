---
datasets:
- venetis/disaster_tweets
language:
- en
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:7613
- loss:BatchAllTripletLoss
widget:
- source_sentence: '?? New Ladies Shoulder Tote #Handbag Faux Leather Hobo Purse Cross
    Body Bag #Womens http://t.co/zujwUiomb3 http://t.co/GBCtmhx7pW'
  sentences:
  - "New Ladies Shoulder Tote Handbag Women Cross Body Bag Faux Leather Fashion Purse\
    \ - Full re\x89Û_ http://t.co/BLAAWHYScT http://t.co/dDR0zjXVQN"
  - Trial Date Set for Man Charged with Arson Burglary http://t.co/WftCrLz32P
  - '@MageAvexis &lt; things. And what if we get attacked?'
- source_sentence: WWI WWII JAPANESE ARMY NAVY MILITARY JAPAN LEATHER WATCH WAR MIDO
    WW1 2 - Full read by eBay http://t.co/QUmcE7W2tY http://t.co/KTKG2sDhHl
  sentences:
  - '#reuters Twelve feared killed in Pakistani air ambulance helicopter crash http://t.co/ShzPyIQok5'
  - "Set our hearts ablaze and every city was a gift And every skyline was like a\
    \ kiss upon the lips @\x89Û_ https://t.co/cYoMPZ1A0Z"
  - They sky was ablaze tonight in Los Angeles. I'm expecting IG and FB to be filled
    with sunset shots if I know my peeps!!
- source_sentence: Attack on Titan game on PS Vita yay! Can't wait for 2016
  sentences:
  - '@SonofBaldwin and he''s the current Nova in the bookslast I checked..he was tied
    into the books in 2011 after Rider died during Annihilation'
  - '@Shayoly yes I love it'
  - Once again black men didn't make it that way. White men did so why are black men
    getting attacked  https://t.co/chkP0GfyNJ
- source_sentence: peanut butter cookie dough blizzard is ??????????????????????
  sentences:
  - 'One Direction Is my pick for http://t.co/q2eBlOKeVE Fan Army #Directioners http://t.co/eNCmhz6y34
    x1402'
  - will there be another jocelyn birthday apocalypse
  - Tracy Blight Thank you for following me!!
- source_sentence: Even though BSG had been sufficiently hyped up for me in all the
    years I somehow delayed watching it I was utterly utterly blown away.
  sentences:
  - 'U.S National Park Services Tonto National Forest: Stop the Annihilation of the
    Salt River Wild Horse... https://t.co/sW1sBua3mN via @Change'
  - Three people died from the heat wave so far
  - "What if every 5000 wins in ranked play gave you a special card back.. Would be\
    \ cool for the long te\x89Û_ http://t.co/vq3yaB2j8N"
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained on the [train](https://huggingface.co/datasets/venetis/disaster_tweets) dataset. It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 tokens
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - [train](https://huggingface.co/datasets/venetis/disaster_tweets)
- **Language:** en
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Even though BSG had been sufficiently hyped up for me in all the years I somehow delayed watching it I was utterly utterly blown away.',
    'What if every 5000 wins in ranked play gave you a special card back.. Would be cool for the long te\x89Û_ http://t.co/vq3yaB2j8N',
    'Three people died from the heat wave so far',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### train

* Dataset: [train](https://huggingface.co/datasets/venetis/disaster_tweets) at [896f918](https://huggingface.co/datasets/venetis/disaster_tweets/tree/896f918ab4d22b0ad24f8e2d01b84acf0742c050)
* Size: 7,613 training samples
* Columns: <code>text</code> and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | text                                                                              | label                                           |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------|
  | type    | string                                                                            | int                                             |
  | details | <ul><li>min: 4 tokens</li><li>mean: 32.39 tokens</li><li>max: 82 tokens</li></ul> | <ul><li>0: ~69.50%</li><li>1: ~30.50%</li></ul> |
* Samples:
  | text                                                                                                                                               | label          |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------|:---------------|
  | <code>Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all</code>                                                                 | <code>1</code> |
  | <code>Forest fire near La Ronge Sask. Canada</code>                                                                                                | <code>1</code> |
  | <code>All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected</code> | <code>1</code> |
* Loss: [<code>BatchAllTripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#batchalltripletloss)

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 256
- `learning_rate`: 1e-05
- `num_train_epochs`: 1000
- `warmup_ratio`: 0.1
- `fp16`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 256
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 1000
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 10.0   | 300   | 4.1926        |
| 20.0   | 600   | 4.1554        |
| 30.0   | 900   | 4.0932        |
| 40.0   | 1200  | 4.0293        |
| 50.0   | 1500  | 3.9737        |
| 60.0   | 1800  | 3.9205        |
| 70.0   | 2100  | 3.8723        |
| 80.0   | 2400  | 3.8295        |
| 90.0   | 2700  | 3.7977        |
| 100.0  | 3000  | 3.7537        |
| 110.0  | 3300  | 3.7179        |
| 120.0  | 3600  | 3.6873        |
| 130.0  | 3900  | 3.6586        |
| 140.0  | 4200  | 3.6374        |
| 150.0  | 4500  | 3.6147        |
| 160.0  | 4800  | 3.595         |
| 170.0  | 5100  | 3.5747        |
| 180.0  | 5400  | 3.5544        |
| 190.0  | 5700  | 3.5353        |
| 200.0  | 6000  | 3.5147        |
| 210.0  | 6300  | 3.5008        |
| 220.0  | 6600  | 3.4852        |
| 230.0  | 6900  | 3.4735        |
| 240.0  | 7200  | 3.4612        |
| 250.0  | 7500  | 3.4474        |
| 260.0  | 7800  | 3.4331        |
| 270.0  | 8100  | 3.4233        |
| 280.0  | 8400  | 3.4141        |
| 290.0  | 8700  | 3.4051        |
| 300.0  | 9000  | 3.3996        |
| 310.0  | 9300  | 3.3912        |
| 320.0  | 9600  | 3.3876        |
| 330.0  | 9900  | 3.3858        |
| 340.0  | 10200 | 3.3801        |
| 350.0  | 10500 | 3.3776        |
| 360.0  | 10800 | 3.3729        |
| 370.0  | 11100 | 3.3699        |
| 380.0  | 11400 | 3.3681        |
| 390.0  | 11700 | 3.362         |
| 400.0  | 12000 | 3.3582        |
| 410.0  | 12300 | 3.3537        |
| 420.0  | 12600 | 3.351         |
| 430.0  | 12900 | 3.3494        |
| 440.0  | 13200 | 3.3458        |
| 450.0  | 13500 | 3.3421        |
| 460.0  | 13800 | 3.3384        |
| 470.0  | 14100 | 3.3382        |
| 480.0  | 14400 | 3.3322        |
| 490.0  | 14700 | 3.3293        |
| 500.0  | 15000 | 3.3266        |
| 510.0  | 15300 | 3.3244        |
| 520.0  | 15600 | 3.3232        |
| 530.0  | 15900 | 3.3198        |
| 540.0  | 16200 | 3.3177        |
| 550.0  | 16500 | 3.3156        |
| 560.0  | 16800 | 3.3155        |
| 570.0  | 17100 | 3.3128        |
| 580.0  | 17400 | 3.3124        |
| 590.0  | 17700 | 3.3113        |
| 600.0  | 18000 | 3.309         |
| 610.0  | 18300 | 3.3077        |
| 620.0  | 18600 | 3.308         |
| 630.0  | 18900 | 3.3052        |
| 640.0  | 19200 | 3.3034        |
| 650.0  | 19500 | 3.3017        |
| 660.0  | 19800 | 3.3013        |
| 670.0  | 20100 | 3.301         |
| 680.0  | 20400 | 3.2988        |
| 690.0  | 20700 | 3.2988        |
| 700.0  | 21000 | 3.297         |
| 710.0  | 21300 | 3.2971        |
| 720.0  | 21600 | 3.2953        |
| 730.0  | 21900 | 3.2941        |
| 740.0  | 22200 | 3.2927        |
| 750.0  | 22500 | 3.2908        |
| 760.0  | 22800 | 3.2923        |
| 770.0  | 23100 | 3.2902        |
| 780.0  | 23400 | 3.2898        |
| 790.0  | 23700 | 3.2886        |
| 800.0  | 24000 | 3.2871        |
| 810.0  | 24300 | 3.2867        |
| 820.0  | 24600 | 3.2859        |
| 830.0  | 24900 | 3.2856        |
| 840.0  | 25200 | 3.2855        |
| 850.0  | 25500 | 3.2854        |
| 860.0  | 25800 | 3.2843        |
| 870.0  | 26100 | 3.2836        |
| 880.0  | 26400 | 3.2834        |
| 890.0  | 26700 | 3.2832        |
| 900.0  | 27000 | 3.2838        |
| 910.0  | 27300 | 3.2828        |
| 920.0  | 27600 | 3.282         |
| 930.0  | 27900 | 3.2813        |
| 940.0  | 28200 | 3.2807        |
| 950.0  | 28500 | 3.2813        |
| 960.0  | 28800 | 3.2805        |
| 970.0  | 29100 | 3.2808        |
| 980.0  | 29400 | 3.2798        |
| 990.0  | 29700 | 3.28          |
| 1000.0 | 30000 | 3.2794        |


### Framework Versions
- Python: 3.11.7
- Sentence Transformers: 3.1.1
- Transformers: 4.45.1
- PyTorch: 2.2.2+cu121
- Accelerate: 0.35.0.dev0
- Datasets: 3.0.1
- Tokenizers: 0.20.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### BatchAllTripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->