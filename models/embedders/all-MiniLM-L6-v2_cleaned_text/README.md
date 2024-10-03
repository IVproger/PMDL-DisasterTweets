---
base_model: sentence-transformers/all-MiniLM-L6-v2
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
- source_sentence: ' attack dimitri arm mageavexi thing and get attack'
  sentences:
  - ' attack attack titan game vita yay cant wait'
  - ' blaze columbus shayoli yes love'
  - ' bodybag  yanke bodi bag mfs'
- source_sentence: ' armi new york wwi wwii japanes armi navi militari japan leather
    watch war mido full read ebay httptcoqumcewti httptcoktkgsdhhl'
  sentences:
  - ' armi one direct pick httptcoqeblokev fan armi direction httptcoencmhzi'
  - ' bleed dmv fashion school ksu hit foot toe bleed'
  - ' bodybag paignton new ladi shoulder tote handbag faux leather hobo purs cross
    bodi bag women httptcozujwuiomb httptcogbctmhxpw'
- source_sentence: ' ambul reuter twelv fear kill pakistani air ambul helicopt crash
    httptcoshzpyiqok'
  sentences:
  - ' annihil nation park servic tonto nation forest stop annihil salt river wild
    hors httpstcommvdspjp via chang'
  - ' three peopl die heat wave far'
  - ' arson trial date set man charg arson burglari httptcowftcrlzp'
- source_sentence: ' annihil connecticut sonofbaldwin hes current nova bookslast checkedh
    tie book rider die annihil'
  sentences:
  - ' annihil swane around annihil damascus syrian armi grind alloosh and his gang
    into the manur pile httptcorakhpbwm'
  - ' ambul happili marri kid ambul sprinter automat frontlin vehicl choic lez compliant
    ebay httptcoevttqpeia'
  - ' apocalyps anoth jocelyn birthday apocalyps'
- source_sentence: ' blownup los angel even though bsg suffici hype year somehow delay
    watch utter utter blown away'
  sentences:
  - ' accid  norwaymfa bahrain polic previous die road accid kill explos httpstcogfjfgtodad'
  - ' blight london willhillbet doubl result live app'
  - ' arsonist america dont anyth nice say come sit'
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 8b3219a92973c328a8e22fadcfa821b5dc75636a -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
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
    ' blownup los angel even though bsg suffici hype year somehow delay watch utter utter blown away',
    ' arsonist america dont anyth nice say come sit',
    ' blight london willhillbet doubl result live app',
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

#### Unnamed Dataset


* Size: 7,613 training samples
* Columns: <code>text</code> and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | text                                                                              | label                                           |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------|
  | type    | string                                                                            | int                                             |
  | details | <ul><li>min: 4 tokens</li><li>mean: 24.59 tokens</li><li>max: 50 tokens</li></ul> | <ul><li>0: ~69.50%</li><li>1: ~30.50%</li></ul> |
* Samples:
  | text                                                                                    | label          |
  |:----------------------------------------------------------------------------------------|:---------------|
  | <code> our deed reason earthquak may allah forgiv</code>                                | <code>1</code> |
  | <code> forest fire near rong sask canada</code>                                         | <code>1</code> |
  | <code> all resid ask shelter place notifi offic evacu shelter place order expect</code> | <code>1</code> |
* Loss: [<code>BatchAllTripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#batchalltripletloss)

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 360
- `learning_rate`: 1e-05
- `num_train_epochs`: 4000
- `warmup_ratio`: 0.1
- `fp16`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 360
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
- `num_train_epochs`: 4000
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
<details><summary>Click to expand</summary>

| Epoch     | Step  | Training Loss |
|:---------:|:-----:|:-------------:|
| 13.6364   | 300   | 4.9899        |
| 27.2727   | 600   | 4.9469        |
| 40.9091   | 900   | 4.6869        |
| 54.5455   | 1200  | 4.4896        |
| 68.1818   | 1500  | 4.377         |
| 81.8182   | 1800  | 4.2955        |
| 95.4545   | 2100  | 4.243         |
| 109.0909  | 2400  | 4.1891        |
| 122.7273  | 2700  | 4.1417        |
| 136.3636  | 3000  | 4.0945        |
| 150.0     | 3300  | 4.0661        |
| 163.6364  | 3600  | 4.0276        |
| 177.2727  | 3900  | 3.9952        |
| 190.9091  | 4200  | 3.9701        |
| 204.5455  | 4500  | 3.9438        |
| 218.1818  | 4800  | 3.921         |
| 231.8182  | 5100  | 3.9033        |
| 245.4545  | 5400  | 3.8654        |
| 259.0909  | 5700  | 3.8452        |
| 272.7273  | 6000  | 3.8162        |
| 286.3636  | 6300  | 3.8054        |
| 300.0     | 6600  | 3.7762        |
| 313.6364  | 6900  | 3.759         |
| 327.2727  | 7200  | 3.7545        |
| 340.9091  | 7500  | 3.7184        |
| 354.5455  | 7800  | 3.7025        |
| 368.1818  | 8100  | 3.6865        |
| 381.8182  | 8400  | 3.6693        |
| 395.4545  | 8700  | 3.644         |
| 409.0909  | 9000  | 3.6292        |
| 422.7273  | 9300  | 3.6053        |
| 436.3636  | 9600  | 3.597         |
| 450.0     | 9900  | 3.5799        |
| 463.6364  | 10200 | 3.5656        |
| 477.2727  | 10500 | 3.5532        |
| 490.9091  | 10800 | 3.5481        |
| 504.5455  | 11100 | 3.5302        |
| 518.1818  | 11400 | 3.5276        |
| 531.8182  | 11700 | 3.5158        |
| 545.4545  | 12000 | 3.5121        |
| 559.0909  | 12300 | 3.515         |
| 572.7273  | 12600 | 3.5004        |
| 586.3636  | 12900 | 3.4951        |
| 600.0     | 13200 | 3.4864        |
| 613.6364  | 13500 | 3.4806        |
| 627.2727  | 13800 | 3.4643        |
| 640.9091  | 14100 | 3.4583        |
| 654.5455  | 14400 | 3.4564        |
| 668.1818  | 14700 | 3.4491        |
| 681.8182  | 15000 | 3.4439        |
| 695.4545  | 15300 | 3.4443        |
| 709.0909  | 15600 | 3.4363        |
| 722.7273  | 15900 | 3.4412        |
| 736.3636  | 16200 | 3.4392        |
| 750.0     | 16500 | 3.4281        |
| 763.6364  | 16800 | 3.4364        |
| 777.2727  | 17100 | 3.428         |
| 790.9091  | 17400 | 3.426         |
| 804.5455  | 17700 | 3.4219        |
| 818.1818  | 18000 | 3.4206        |
| 831.8182  | 18300 | 3.4146        |
| 845.4545  | 18600 | 3.4177        |
| 859.0909  | 18900 | 3.4153        |
| 872.7273  | 19200 | 3.4092        |
| 886.3636  | 19500 | 3.4086        |
| 900.0     | 19800 | 3.401         |
| 913.6364  | 20100 | 3.4088        |
| 927.2727  | 20400 | 3.3956        |
| 940.9091  | 20700 | 3.4041        |
| 954.5455  | 21000 | 3.3944        |
| 968.1818  | 21300 | 3.3908        |
| 981.8182  | 21600 | 3.3919        |
| 995.4545  | 21900 | 3.3908        |
| 1009.0909 | 22200 | 3.3876        |
| 1022.7273 | 22500 | 3.3929        |
| 1036.3636 | 22800 | 3.3924        |
| 1050.0    | 23100 | 3.3842        |
| 1063.6364 | 23400 | 3.3841        |
| 1077.2727 | 23700 | 3.3806        |
| 1090.9091 | 24000 | 3.3761        |
| 1104.5455 | 24300 | 3.3766        |
| 1118.1818 | 24600 | 3.3766        |
| 1131.8182 | 24900 | 3.3746        |
| 1145.4545 | 25200 | 3.3785        |
| 1159.0909 | 25500 | 3.3828        |
| 1172.7273 | 25800 | 3.3763        |
| 1186.3636 | 26100 | 3.3753        |
| 1200.0    | 26400 | 3.3776        |
| 1213.6364 | 26700 | 3.3771        |
| 1227.2727 | 27000 | 3.3736        |
| 1240.9091 | 27300 | 3.3773        |
| 1254.5455 | 27600 | 3.3712        |
| 1268.1818 | 27900 | 3.3753        |
| 1281.8182 | 28200 | 3.3772        |
| 1295.4545 | 28500 | 3.3763        |
| 1309.0909 | 28800 | 3.3754        |
| 1322.7273 | 29100 | 3.3762        |
| 1336.3636 | 29400 | 3.3748        |
| 1350.0    | 29700 | 3.37          |
| 1363.6364 | 30000 | 3.3667        |
| 1377.2727 | 30300 | 3.3732        |
| 1390.9091 | 30600 | 3.3645        |
| 1404.5455 | 30900 | 3.3669        |
| 1418.1818 | 31200 | 3.3616        |
| 1431.8182 | 31500 | 3.3646        |
| 1445.4545 | 31800 | 3.3642        |
| 1459.0909 | 32100 | 3.3717        |
| 1472.7273 | 32400 | 3.3585        |
| 1486.3636 | 32700 | 3.3634        |
| 1500.0    | 33000 | 3.359         |
| 1513.6364 | 33300 | 3.3604        |
| 1527.2727 | 33600 | 3.3644        |
| 1540.9091 | 33900 | 3.3605        |
| 1554.5455 | 34200 | 3.363         |
| 1568.1818 | 34500 | 3.3579        |
| 1581.8182 | 34800 | 3.359         |
| 1595.4545 | 35100 | 3.3643        |
| 1609.0909 | 35400 | 3.3579        |
| 1622.7273 | 35700 | 3.3568        |
| 1636.3636 | 36000 | 3.3563        |
| 1650.0    | 36300 | 3.3579        |
| 1663.6364 | 36600 | 3.36          |
| 1677.2727 | 36900 | 3.3577        |
| 1690.9091 | 37200 | 3.3604        |
| 1704.5455 | 37500 | 3.353         |
| 1718.1818 | 37800 | 3.3575        |
| 1731.8182 | 38100 | 3.35          |
| 1745.4545 | 38400 | 3.3473        |
| 1759.0909 | 38700 | 3.3492        |
| 1772.7273 | 39000 | 3.35          |
| 1786.3636 | 39300 | 3.3487        |
| 1800.0    | 39600 | 3.3582        |
| 1813.6364 | 39900 | 3.3499        |
| 1827.2727 | 40200 | 3.354         |
| 1840.9091 | 40500 | 3.3567        |
| 1854.5455 | 40800 | 3.3466        |
| 1868.1818 | 41100 | 3.3468        |
| 1881.8182 | 41400 | 3.3597        |
| 1895.4545 | 41700 | 3.344         |
| 1909.0909 | 42000 | 3.3503        |
| 1922.7273 | 42300 | 3.3498        |
| 1936.3636 | 42600 | 3.3441        |
| 1950.0    | 42900 | 3.348         |
| 1963.6364 | 43200 | 3.3442        |
| 1977.2727 | 43500 | 3.3488        |
| 1990.9091 | 43800 | 3.3452        |
| 2004.5455 | 44100 | 3.3437        |
| 2018.1818 | 44400 | 3.3473        |
| 2031.8182 | 44700 | 3.3377        |
| 2045.4545 | 45000 | 3.3414        |
| 2059.0909 | 45300 | 3.3428        |
| 2072.7273 | 45600 | 3.3413        |
| 2086.3636 | 45900 | 3.3445        |
| 2100.0    | 46200 | 3.3397        |
| 2113.6364 | 46500 | 3.3422        |
| 2127.2727 | 46800 | 3.343         |
| 2140.9091 | 47100 | 3.3457        |
| 2154.5455 | 47400 | 3.344         |
| 2168.1818 | 47700 | 3.3436        |
| 2181.8182 | 48000 | 3.3452        |
| 2195.4545 | 48300 | 3.3443        |
| 2209.0909 | 48600 | 3.3431        |
| 2222.7273 | 48900 | 3.3421        |
| 2236.3636 | 49200 | 3.3421        |
| 2250.0    | 49500 | 3.3462        |
| 2263.6364 | 49800 | 3.3446        |
| 2277.2727 | 50100 | 3.3451        |
| 2290.9091 | 50400 | 3.3414        |
| 2304.5455 | 50700 | 3.3396        |
| 2318.1818 | 51000 | 3.3401        |
| 2331.8182 | 51300 | 3.3421        |
| 2345.4545 | 51600 | 3.3408        |
| 2359.0909 | 51900 | 3.344         |
| 2372.7273 | 52200 | 3.3403        |
| 2386.3636 | 52500 | 3.3417        |
| 2400.0    | 52800 | 3.3412        |
| 2413.6364 | 53100 | 3.3409        |
| 2427.2727 | 53400 | 3.3382        |
| 2440.9091 | 53700 | 3.3385        |
| 2454.5455 | 54000 | 3.3382        |
| 2468.1818 | 54300 | 3.3346        |
| 2481.8182 | 54600 | 3.3357        |
| 2495.4545 | 54900 | 3.3381        |
| 2509.0909 | 55200 | 3.3391        |
| 2522.7273 | 55500 | 3.3335        |
| 2536.3636 | 55800 | 3.3371        |
| 2550.0    | 56100 | 3.3398        |
| 2563.6364 | 56400 | 3.3377        |
| 2577.2727 | 56700 | 3.3378        |
| 2590.9091 | 57000 | 3.3413        |
| 2604.5455 | 57300 | 3.3408        |
| 2618.1818 | 57600 | 3.3358        |
| 2631.8182 | 57900 | 3.3364        |
| 2645.4545 | 58200 | 3.3379        |
| 2659.0909 | 58500 | 3.3366        |
| 2672.7273 | 58800 | 3.3357        |
| 2686.3636 | 59100 | 3.3345        |
| 2700.0    | 59400 | 3.3339        |
| 2713.6364 | 59700 | 3.3362        |
| 2727.2727 | 60000 | 3.3394        |
| 2740.9091 | 60300 | 3.3361        |
| 2754.5455 | 60600 | 3.3347        |
| 2768.1818 | 60900 | 3.3357        |
| 2781.8182 | 61200 | 3.3375        |
| 2795.4545 | 61500 | 3.3324        |
| 2809.0909 | 61800 | 3.3392        |
| 2822.7273 | 62100 | 3.3383        |
| 2836.3636 | 62400 | 3.333         |
| 2850.0    | 62700 | 3.3357        |
| 2863.6364 | 63000 | 3.3337        |
| 2877.2727 | 63300 | 3.3338        |
| 2890.9091 | 63600 | 3.335         |
| 2904.5455 | 63900 | 3.3373        |
| 2918.1818 | 64200 | 3.337         |
| 2931.8182 | 64500 | 3.3315        |
| 2945.4545 | 64800 | 3.3316        |
| 2959.0909 | 65100 | 3.3309        |
| 2972.7273 | 65400 | 3.3318        |
| 2986.3636 | 65700 | 3.3314        |
| 3000.0    | 66000 | 3.3348        |
| 3013.6364 | 66300 | 3.3378        |
| 3027.2727 | 66600 | 3.3323        |
| 3040.9091 | 66900 | 3.3317        |
| 3054.5455 | 67200 | 3.329         |
| 3068.1818 | 67500 | 3.3307        |
| 3081.8182 | 67800 | 3.3292        |
| 3095.4545 | 68100 | 3.3324        |
| 3109.0909 | 68400 | 3.3311        |
| 3122.7273 | 68700 | 3.3323        |
| 3136.3636 | 69000 | 3.3309        |
| 3150.0    | 69300 | 3.3324        |
| 3163.6364 | 69600 | 3.3293        |
| 3177.2727 | 69900 | 3.3297        |
| 3190.9091 | 70200 | 3.328         |
| 3204.5455 | 70500 | 3.3286        |
| 3218.1818 | 70800 | 3.3286        |
| 3231.8182 | 71100 | 3.3308        |
| 3245.4545 | 71400 | 3.3245        |
| 3259.0909 | 71700 | 3.328         |
| 3272.7273 | 72000 | 3.331         |
| 3286.3636 | 72300 | 3.3234        |
| 3300.0    | 72600 | 3.3252        |
| 3313.6364 | 72900 | 3.334         |
| 3327.2727 | 73200 | 3.3254        |
| 3340.9091 | 73500 | 3.3315        |
| 3354.5455 | 73800 | 3.3229        |
| 3368.1818 | 74100 | 3.3275        |
| 3381.8182 | 74400 | 3.3276        |
| 3395.4545 | 74700 | 3.3278        |
| 3409.0909 | 75000 | 3.3245        |
| 3422.7273 | 75300 | 3.3255        |
| 3436.3636 | 75600 | 3.3281        |
| 3450.0    | 75900 | 3.3255        |
| 3463.6364 | 76200 | 3.3306        |
| 3477.2727 | 76500 | 3.3274        |
| 3490.9091 | 76800 | 3.3291        |
| 3504.5455 | 77100 | 3.3277        |
| 3518.1818 | 77400 | 3.3313        |
| 3531.8182 | 77700 | 3.3264        |
| 3545.4545 | 78000 | 3.3282        |
| 3559.0909 | 78300 | 3.3282        |
| 3572.7273 | 78600 | 3.3301        |
| 3586.3636 | 78900 | 3.3277        |
| 3600.0    | 79200 | 3.3288        |
| 3613.6364 | 79500 | 3.3287        |
| 3627.2727 | 79800 | 3.3231        |
| 3640.9091 | 80100 | 3.3243        |
| 3654.5455 | 80400 | 3.3243        |
| 3668.1818 | 80700 | 3.3239        |
| 3681.8182 | 81000 | 3.3283        |
| 3695.4545 | 81300 | 3.3272        |
| 3709.0909 | 81600 | 3.3267        |
| 3722.7273 | 81900 | 3.3262        |
| 3736.3636 | 82200 | 3.3272        |
| 3750.0    | 82500 | 3.3265        |
| 3763.6364 | 82800 | 3.3275        |
| 3777.2727 | 83100 | 3.3261        |
| 3790.9091 | 83400 | 3.3267        |
| 3804.5455 | 83700 | 3.3292        |
| 3818.1818 | 84000 | 3.3264        |
| 3831.8182 | 84300 | 3.3256        |
| 3845.4545 | 84600 | 3.3287        |
| 3859.0909 | 84900 | 3.3273        |
| 3872.7273 | 85200 | 3.329         |
| 3886.3636 | 85500 | 3.3281        |
| 3900.0    | 85800 | 3.3276        |
| 3913.6364 | 86100 | 3.3277        |
| 3927.2727 | 86400 | 3.3286        |
| 3940.9091 | 86700 | 3.3266        |
| 3954.5455 | 87000 | 3.3261        |
| 3968.1818 | 87300 | 3.3308        |
| 3981.8182 | 87600 | 3.3275        |
| 3995.4545 | 87900 | 3.3219        |

</details>

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