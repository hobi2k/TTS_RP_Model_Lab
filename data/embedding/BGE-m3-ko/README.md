---
language:
- ko
- en
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
- dot_accuracy@1
- dot_accuracy@3
- dot_accuracy@5
- dot_accuracy@10
- dot_precision@1
- dot_precision@3
- dot_precision@5
- dot_precision@10
- dot_recall@1
- dot_recall@3
- dot_recall@5
- dot_recall@10
- dot_ndcg@10
- dot_mrr@10
- dot_map@100
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
widget:
- source_sentence: 대한지적공사 관계자는 "오랜 진통 끝에 지적재조사사업을 추진하게 돼 기쁘다"면서도 뭐라고 말했어?
  sentences:
  - >-
    2018 평창 동계올림픽이 개막하기 전 '공공의 적'은 영하 10도를 넘는 추위였다. 개막을 즈음해 추위는 조금 수그러드는가 싶더니
    바람이 멈추지 않아 대회 2일 차부터 경기가 잇달아 연기·취소됐다.

    올림픽 조직위원회와 국제스키연맹(FIS)은 11일 오전 11시 정선 알파인 경기장에서 열릴 예정이던 알파인 스키 남자 활강 경기를
    강풍으로 연기하기로 했다고 밝혔다. FIS는 “강풍이 경기장에 하루 종일 계속 불 것으로 전망돼 일정을 연기했다”고 밝혔다. 조직위는
    연기된 남자 활강 경기를 오는 15일 오전 11시에 치르고, 이 시간대에 원래 열릴 예정이던 남자 슈퍼대회전 경기 시간을 하루 뒤인
    16일 오전 11시로 순연하기로 했다.

    이어 이날 오후 1시30분부터 열릴 예정이던 스노보드 여자 슬로프스타일 예선 경기는 연기를 거듭하다 취소됐다. 조직위는 예선 없이 다음
    날 결선에서 참가자 27명이 한번에 경기해 순위를 가리기로 했다.

    강풍이 경기 진행에 영향을 미칠 것이란 예상은 대회 전부터 있었다. 올림픽 대회 슬로프가 설치된 정선·용평 알파인 경기장과 휘닉스 스노
    경기장은 슬로프 상단부의 해발고도가 900m가 넘는다. 임장호 조직위 기상기후팀장은 “알파인 스키는 상단부에 강한 바람이 불면, 선수들을
    실어나르는 곤돌라를 움직이기 어렵다”며 “스노보드나 프리스타일 스키는 순간적인 돌풍이 불 때 선수들이 다칠 가능성도 있다”고 말했다.

    바람이 경기에 미치는 영향을 알기에 조직위도 강풍을 비롯한 5가지 긴급 기상 상황을 가정해 경기 운영 매뉴얼을 만들었다. 이날 경기
    취소도 매뉴얼에 따른 조치였다. 임 팀장은 “12~13일 바람이 잦아들다가 14일에 다시 강풍이 불겠지만, 15일부터는 다시 잦아들
    것으로 보고 있다”며 “향후 강풍으로 경기가 연기돼도 올림픽 폐막 전 최대한 모든 경기를 끝내려 하고 있다”고 했다. 다만 경기 일정이
    바뀌면 참가 선수들과 코칭스태프가 어떻게 컨디션을 조절하며 경기를 준비할지 깊은 고민에 빠질 것으로 보인다.
  - >-
    지적도면과 실제 경계가 맞지 않는 '지적불부합지'에 대한 재조사가 실시된다. 국토해양부는 지적도상 경계와 실제 경계가 일치하지 않는
    지적불부합지에 대해 2030년까지 지적재조사를 추진한다고 지난달 30일 밝혔다. 이와 관련 김기현 의원이 대표발의한 지적재조사특별법안이
    이날 국회 상임위를 통과했다. 지적불부합지는 경계분쟁과 민원의 대상이 되고 있는데, 현재 전체 필지의 약 15%(554만필지)에 이를
    것으로 추정된다. 특히 상당수는 지적측량이 불가능해 소유권 이전이나 건축행위 등 재산권 행사가 불가능하거나 제한받고 있어 조정이 시급한
    상황이다. 이에 따라 1995년 지적재조사사업추진 기본계획이 수립되고, 이듬해 지적재조사특별법이 입법예고됐지만 관련 부처들의 반대로
    무산됐다. 이후 2000년 다시 재조사사업 기본계획이 수립되고, 2006년 토지조사특별법안이 제출됐으나 성사되지 못한 채 오늘에 이르고
    있다. 지적불부합지는 100년 전 낙후된 기술로 만든 종이지적을 계속 사용하면서 종이도면의 신축, 경계선의 굵기, 개인오차 등으로
    생겨났다. 또 대장이 토지·임야대장으로 이원화돼 있고, 도면도 7종의 축척으로 등록된 것도 원인으로 꼽힌다. 일례로 1:1200 축척의
    압구정동 대지(280㎡, 1000만원/㎡)의 경우 지적도상 경계가 0.8mm 오차가 나면 실제 면적에선 27㎡의 차이가 발생, 약
    2억7000만원의 땅값이 차이나게 된다. 6·25전쟁으로 전국 106만1000필지의 지적공부가 분·소실되고, 약 80%의 지적측량기준점을
    잃어버린 것도 한 원인이다. 토지공법학회는 2005년 지적불부합에 따른 경계분쟁으로 연간 약 3800억원의 소송비용이 발생한 것으로
    추정했다. 또 경계확인측량으로 연간 900억원의 비용이 지출되고 있다. 정부는 총 8410억원을 투입, 2020년까지 280만필지를,
    나머지 274만필지는 2030년까지 정비할 계획이다. 국토부 관계자는 "지적불부합지가 정비되면 경계분쟁이 해소돼 사회적 비용을 절감할 수
    있고, 개인의 재산권 행사도 수월해 질 것"이라고 기대했다. 그러나 전국에 걸친 전면적인 지적재조사가 아니라 불부합지를 중심으로 한
    단계적 추진이어서 한계가 있다는 지적이다. 앞으로 재조사가 진행되면 불부합지가 계속 나타나게 될 것인데 그 때마다 경계조정을 해야 하는
    번거로움이 있다는 것. 특히 불부합지에 대한 경계조정은 이해가 첨예하게 충돌하다 보니 사업추진이 매우 어렵다. 이 때문에 전면적인
    재조사를 통해 한 번에 마무리하는 것이 수월하다는 설명이다. 대한지적공사 관계자는 "오랜 진통 끝에 지적재조사사업을 추진하게 돼
    기쁘다"면서도 "원래 전면적인 사업추진을 원했으나 예산 등의 문제로 단계적으로 진행하게 돼 아쉽다"고 말했다.
model-index:
- name: SentenceTransformer
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: miracl
      type: miracl
    metrics:
    - type: cosine_accuracy@1
      value: 0.6103286384976526
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.8169014084507042
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.8732394366197183
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.92018779342723
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.6103286384976526
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.378716744913928
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.27605633802816903
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.17276995305164322
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.3846655691726114
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.5901991071005155
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.6794216477315068
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.7694903427297795
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.6833112035481234
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.7262426410313736
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.6073885234240499
      name: Cosine Map@100
    - type: dot_accuracy@1
      value: 0.6103286384976526
      name: Dot Accuracy@1
    - type: dot_accuracy@3
      value: 0.8169014084507042
      name: Dot Accuracy@3
    - type: dot_accuracy@5
      value: 0.8732394366197183
      name: Dot Accuracy@5
    - type: dot_accuracy@10
      value: 0.92018779342723
      name: Dot Accuracy@10
    - type: dot_precision@1
      value: 0.6103286384976526
      name: Dot Precision@1
    - type: dot_precision@3
      value: 0.378716744913928
      name: Dot Precision@3
    - type: dot_precision@5
      value: 0.27605633802816903
      name: Dot Precision@5
    - type: dot_precision@10
      value: 0.17276995305164322
      name: Dot Precision@10
    - type: dot_recall@1
      value: 0.3846655691726114
      name: Dot Recall@1
    - type: dot_recall@3
      value: 0.5901991071005155
      name: Dot Recall@3
    - type: dot_recall@5
      value: 0.6794216477315068
      name: Dot Recall@5
    - type: dot_recall@10
      value: 0.7694903427297795
      name: Dot Recall@10
    - type: dot_ndcg@10
      value: 0.6723275985412543
      name: Dot Ndcg@10
    - type: dot_mrr@10
      value: 0.7262426410313736
      name: Dot Mrr@10
    - type: dot_map@100
      value: 0.6073885234240499
      name: Dot Map@100
license: apache-2.0
base_model:
- BAAI/bge-m3
---


<img src="https://cdn-uploads.huggingface.co/production/uploads/642b0c2fecec03b4464a1d9b/9uN5ypGY-GRGgakLs_s1o.png" width="600">


# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained on the train_set dataset. It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

- Learning other languages ​​besides Chinese and English is insufficient, so additional learning is needed to optimize use of other languages.
- This model is additionally trained on the Korean dataset.

### Model Description
- **Model Type:** Sentence Transformer
  Transformer Encoder
- **Maximum Sequence Length:** 8192 tokens
- **Output Dimensionality:** 1024 tokens
- **Similarity Function:** Cosine Similarity

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 8192, 'do_lower_case': False}) with Transformer model: XLMRobertaModel 
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
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
model = SentenceTransformer("dragonkue/bge-m3-ko")
# Run inference
sentences = [
    '수급권자 중 근로 능력이 없는 임산부는 몇 종에 해당하니?',
    '내년부터 저소득층 1세 미만 아동의 \n의료비 부담이 더 낮아진다!\n의료급여제도 개요\n□ (목적) 생활유지 능력이 없거나 생활이 어려운 국민들에게 발생하는 질병, 부상, 출산 등에 대해 국가가 의료서비스 제공\n□ (지원대상) 국민기초생활보장 수급권자, 타 법에 의한 수급권자 등\n\n| 구분 | 국민기초생활보장법에 의한 수급권자 | 국민기초생활보장법 이외의 타 법에 의한 수급권자 |\n| --- | --- | --- |\n| 1종 | ○ 국민기초생활보장 수급권자 중 근로능력이 없는 자만으로 구성된 가구 - 18세 미만, 65세 이상 - 4급 이내 장애인 - 임산부, 병역의무이행자 등 | ○ 이재민(재해구호법) ○ 의상자 및 의사자의 유족○ 국내 입양된 18세 미만 아동○ 국가유공자 및 그 유족․가족○ 국가무형문화재 보유자 및 그 가족○ 새터민(북한이탈주민)과 그 가족○ 5․18 민주화운동 관련자 및 그 유가족○ 노숙인 ※ 행려환자 (의료급여법 시행령) |\n| 2종 | ○ 국민기초생활보장 수급권자 중 근로능력이 있는 가구 | - |\n',
    '이어 이날 오후 1시30분부터 열릴 예정이던 스노보드 여자 슬로프스타일 예선 경기는 연기를 거듭하다 취소됐다. 조직위는 예선 없이 다음 날 결선에서 참가자 27명이 한번에 경기해 순위를 가리기로 했다.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

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

## Evaluation

### Metrics
- ndcg, mrr, map metrics are metrics that consider ranking, while accuracy, precision, and recall are metrics that do not consider ranking. (Example: When considering ranking for retrieval top 10, different scores are given when the correct document is in 1st place and when it is in 10th place. However, accuracy, precision, and recall scores are the same if they are in the top 10.)

#### Information Retrieval
* Korean Embedding Benchmark is a benchmark with a relatively long 3/4 quantile of string length of 1024

##### Korean Embedding Benchmark with AutoRAG

This is a benchmark of Korean embedding models. 
(https://github.com/Marker-Inc-Korea/AutoRAG-example-korean-embedding-benchmark)

- Top-k 1

| Model name                            | F1         | Recall     | Precision  | mAP        | mRR        | NDCG       |
|---------------------------------------|------------|------------|------------|------------|------------|------------|
| paraphrase-multilingual-mpnet-base-v2 | 0.3596     | 0.3596     | 0.3596     | 0.3596     | 0.3596     | 0.3596     |
| KoSimCSE-roberta                      | 0.4298     | 0.4298     | 0.4298     | 0.4298     | 0.4298     | 0.4298     |
| Cohere embed-multilingual-v3.0        | 0.3596     | 0.3596     | 0.3596     | 0.3596     | 0.3596     | 0.3596     |
| openai ada 002                        | 0.4737     | 0.4737     | 0.4737     | 0.4737     | 0.4737     | 0.4737     |
| multilingual-e5-large-instruct        | 0.4649     | 0.4649     | 0.4649     | 0.4649     | 0.4649     | 0.4649     |
| Upstage Embedding                     | 0.6579     | 0.6579     | 0.6579     | 0.6579     | 0.6579     | 0.6579     |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.2982     | 0.2982     | 0.2982     | 0.2982     | 0.2982     | 0.2982     |
| openai_embed_3_small                  | 0.5439     | 0.5439     | 0.5439     | 0.5439     | 0.5439     | 0.5439     |
| ko-sroberta-multitask                 | 0.4211     | 0.4211     | 0.4211     | 0.4211     | 0.4211     | 0.4211     |
| openai_embed_3_large                  | 0.6053     | 0.6053     | 0.6053     | 0.6053     | 0.6053     | 0.6053     |
| KU-HIAI-ONTHEIT-large-v1              | 0.7105     | 0.7105     | 0.7105     | 0.7105     | 0.7105     | 0.7105     |
| KU-HIAI-ONTHEIT-large-v1.1            | 0.7193     | 0.7193     | 0.7193     | 0.7193     | 0.7193     | 0.7193     |
| kf-deberta-multitask                  | 0.4561     | 0.4561     | 0.4561     | 0.4561     | 0.4561     | 0.4561     |
| gte-multilingual-base                 | 0.5877     | 0.5877     | 0.5877     | 0.5877     | 0.5877     | 0.5877     |
| KoE5                                  | 0.7018     | 0.7018     | 0.7018     | 0.7018     | 0.7018     | 0.7018     |
| BGE-m3                                | 0.6578     | 0.6578     | 0.6578     | 0.6578     | 0.6578     | 0.6578     |
| bge-m3-korean                         | 0.5351     | 0.5351     | 0.5351     | 0.5351     | 0.5351     | 0.5351     |
| **BGE-m3-ko**                         | **0.7456** | **0.7456** | **0.7456** | **0.7456** | **0.7456** | **0.7456** |

- Top-k 3

| Model name                            | F1         | Recall     | Precision  | mAP        | mRR        | NDCG       |
|---------------------------------------|------------|------------|------------|------------|------------|------------|
| paraphrase-multilingual-mpnet-base-v2 | 0.2368     | 0.4737     | 0.1579     | 0.2032     | 0.2032     | 0.2712     |
| KoSimCSE-roberta                      | 0.3026     | 0.6053     | 0.2018     | 0.2661     | 0.2661     | 0.3515     |
| Cohere embed-multilingual-v3.0        | 0.2851     | 0.5702     | 0.1901     | 0.2515     | 0.2515     | 0.3321     |
| openai ada 002                        | 0.3553     | 0.7105     | 0.2368     | 0.3202     | 0.3202     | 0.4186     |
| multilingual-e5-large-instruct        | 0.3333     | 0.6667     | 0.2222     | 0.2909     | 0.2909     | 0.3856     |
| Upstage Embedding                     | 0.4211     | 0.8421     | 0.2807     | **0.3509** | **0.3509** | 0.4743     |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.2061     | 0.4123     | 0.1374     | 0.1740     | 0.1740     | 0.2340     |
| openai_embed_3_small                  | 0.3640     | 0.7281     | 0.2427     | 0.3026     | 0.3026     | 0.4097     |
| ko-sroberta-multitask                 | 0.2939     | 0.5877     | 0.1959     | 0.2500     | 0.2500     | 0.3351     |
| openai_embed_3_large                  | 0.3947     | 0.7895     | 0.2632     | 0.3348     | 0.3348     | 0.4491     |
| KU-HIAI-ONTHEIT-large-v1              | 0.4386     | 0.8772     | 0.2924     | 0.3421     | 0.3421     | 0.4766     |
| KU-HIAI-ONTHEIT-large-v1.1            | 0.4430     | 0.8860     | 0.2953     | 0.3406     | 0.3406     | 0.4778     |
| kf-deberta-multitask                  | 0.3158     | 0.6316     | 0.2105     | 0.2792     | 0.2792     | 0.3679     |
| gte-multilingual-base                 | 0.4035     | 0.8070     | 0.2690     | 0.3450     | 0.3450     | 0.4614     |
| KoE5                                  | 0.4254     | 0.8509     | 0.2836     | 0.3173     | 0.3173     | 0.4514     |
| BGE-m3                                | 0.4254     | 0.8508     | 0.2836     | 0.3421     | 0.3421     | 0.4701     |
| bge-m3-korean                         | 0.3684     | 0.7368     | 0.2456     | 0.3143     | 0.3143     | 0.4207     | 
| **BGE-m3-ko**                         | **0.4517** | **0.9035** | **0.3011** | 0.3494     | 0.3494     | **0.4886** |

- Top-k 5

| Model name                            | F1         | Recall     | Precision  | mAP        | mRR        | NDCG       |
|---------------------------------------|------------|------------|------------|------------|------------|------------|
| paraphrase-multilingual-mpnet-base-v2 | 0.1813     | 0.5439     | 0.1088     | 0.1575     | 0.1575     | 0.2491     |
| KoSimCSE-roberta                      | 0.2164     | 0.6491     | 0.1298     | 0.1751     | 0.1751     | 0.2873     |
| Cohere embed-multilingual-v3.0        | 0.2076     | 0.6228     | 0.1246     | 0.1640     | 0.1640     | 0.2731     |
| openai ada 002                        | 0.2602     | 0.7807     | 0.1561     | 0.2139     | 0.2139     | 0.3486     |
| multilingual-e5-large-instruct        | 0.2544     | 0.7632     | 0.1526     | 0.2194     | 0.2194     | 0.3487     |
| Upstage Embedding                     | 0.2982     | 0.8947     | 0.1789     | **0.2237** | **0.2237** | 0.3822     |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.1637     | 0.4912     | 0.0982     | 0.1437     | 0.1437     | 0.2264     |
| openai_embed_3_small                  | 0.2690     | 0.8070     | 0.1614     | 0.2148     | 0.2148     | 0.3553     |
| ko-sroberta-multitask                 | 0.2164     | 0.6491     | 0.1298     | 0.1697     | 0.1697     | 0.2835     |
| openai_embed_3_large                  | 0.2807     | 0.8421     | 0.1684     | 0.2088     | 0.2088     | 0.3586     |
| KU-HIAI-ONTHEIT-large-v1              | 0.3041     | 0.9123     | 0.1825     | 0.2137     | 0.2137     | 0.3783     |
| KU-HIAI-ONTHEIT-large-v1.1            | **0.3099** | **0.9298** | **0.1860** | 0.2148     | 0.2148     | **0.3834** |
| kf-deberta-multitask                  | 0.2281     | 0.6842     | 0.1368     | 0.1724     | 0.1724     | 0.2939     |
| gte-multilingual-base                 | 0.2865     | 0.8596     | 0.1719     | 0.2096     | 0.2096     | 0.3637     |
| KoE5                                  | 0.2982     | 0.8947     | 0.1789     | 0.2054     | 0.2054     | 0.3678     |
| BGE-m3                                | 0.3041     | 0.9123     | 0.1825     | 0.2193     | 0.2193     | 0.3832     |
| bge-m3-korean                         | 0.2661     | 0.7982     | 0.1596     | 0.2116     | 0.2116     | 0.3504     |
| **BGE-m3-ko**                         | **0.3099** | **0.9298** | **0.1860** | 0.2098     | 0.2098     | 0.3793     |

- Top-k 10

| Model name                            | F1         | Recall     | Precision  | mAP        | mRR        | NDCG       |
|---------------------------------------|------------|------------|------------|------------|------------|------------|
| paraphrase-multilingual-mpnet-base-v2 | 0.1212     | 0.6667     | 0.0667     | **0.1197** | **0.1197** | 0.2382     |
| KoSimCSE-roberta                      | 0.1324     | 0.7281     | 0.0728     | 0.1080     | 0.1080     | 0.2411     |
| Cohere embed-multilingual-v3.0        | 0.1324     | 0.7281     | 0.0728     | 0.1150     | 0.1150     | 0.2473     |
| openai ada 002                        | 0.1563     | 0.8596     | 0.0860     | 0.1051     | 0.1051     | 0.2673     |
| multilingual-e5-large-instruct        | 0.1483     | 0.8158     | 0.0816     | 0.0980     | 0.0980     | 0.2520     |
| Upstage Embedding                     | 0.1707     | 0.9386     | 0.0939     | 0.1078     | 0.1078     | 0.2848     |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.1053     | 0.5789     | 0.0579     | 0.0961     | 0.0961     | 0.2006     |
| openai_embed_3_small                  | 0.1547     | 0.8509     | 0.0851     | 0.0984     | 0.0984     | 0.2593     |
| ko-sroberta-multitask                 | 0.1276     | 0.7018     | 0.0702     | 0.0986     | 0.0986     | 0.2275     |
| openai_embed_3_large                  | 0.1643     | 0.9035     | 0.0904     | 0.1180     | 0.1180     | 0.2855     |
| KU-HIAI-ONTHEIT-large-v1              | 0.1707     | 0.9386     | 0.0939     | 0.1105     | 0.1105     | 0.2860     |
| KU-HIAI-ONTHEIT-large-v1.1            | 0.1722     | 0.9474     | 0.0947     | 0.1033     | 0.1033     | 0.2822     |
| kf-deberta-multitask                  | 0.1388     | 0.7632     | 0.0763     | 0.1        | 0.1        | 0.2422     |
| gte-multilingual-base                 | 0.1675     | 0.9211     | 0.0921     | 0.1066     | 0.1066     | 0.2805     |
| KoE5                                  | 0.1675     | 0.9211     | 0.0921     | 0.1011     | 0.1011     | 0.2750     |
| BGE-m3                                | 0.1707     | 0.9386     | 0.0939     | 0.1130     | 0.1130     | 0.2884     |
| bge-m3-korean                         | 0.1579     | 0.8684     | 0.0868     | 0.1093     | 0.1093     | 0.2721     |
| **BGE-m3-ko**                         | **0.1770** | **0.9736** | **0.0974** | 0.1097     | 0.1097     | **0.2932** |



#### Information Retrieval
* Dataset: `miracl-ko` (https://github.com/project-miracl/miracl)
* miracl benchmark is a benchmark with a relatively short 3/4 quantile of string length of 220 on the Korean Wikidata set.
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.6103     |
| cosine_accuracy@3   | 0.8169     |
| cosine_accuracy@5   | 0.8732     |
| cosine_accuracy@10  | 0.9202     |
| cosine_precision@1  | 0.6103     |
| cosine_precision@3  | 0.3787     |
| cosine_precision@5  | 0.2761     |
| cosine_precision@10 | 0.1728     |
| cosine_recall@1     | 0.3847     |
| cosine_recall@3     | 0.5902     |
| cosine_recall@5     | 0.6794     |
| cosine_recall@10    | 0.7695     |
| **cosine_ndcg@10**  | **0.6833** |
| cosine_mrr@10       | 0.7262     |
| cosine_map@100      | 0.6074     |
| dot_accuracy@1      | 0.6103     |
| dot_accuracy@3      | 0.8169     |
| dot_accuracy@5      | 0.8732     |
| dot_accuracy@10     | 0.9202     |
| dot_precision@1     | 0.6103     |
| dot_precision@3     | 0.3787     |
| dot_precision@5     | 0.2761     |
| dot_precision@10    | 0.1728     |
| dot_recall@1        | 0.3847     |
| dot_recall@3        | 0.5902     |
| dot_recall@5        | 0.6794     |
| dot_recall@10       | 0.7695     |
| dot_ndcg@10         | 0.6723     |
| dot_mrr@10          | 0.7262     |
| dot_map@100         | 0.6074     |

## Bias, Risks and Limitations

- Since the evaluation results are different for each domain, it is necessary to compare and evaluate the model in your own domain. In the Miracl benchmark, the evaluation was conducted using the Korean Wikipedia as a corpus, and in this case, the cosine_ndcg@10 score dropped by 0.02 points after learning. However, in the Auto-RAG benchmark, which is a financial domain, the ndcg score increased by 0.09 when it was top 1. This model may be advantageous for use in a specific domain.
- Also, since the miracl benchmark consists of a corpus of relatively short strings, while the Korean Embedding Benchmark consists of a corpus of longer strings, this model may be more advantageous if the length of the corpus you want to use is long.


### Training Hyperparameters
#### Non-Default Hyperparameters
The batch size was referenced from the following paper: Text Embeddings by Weakly-Supervised Contrastive Pre-training (https://arxiv.org/pdf/2212.03533)

- `eval_strategy`: steps
- `per_device_train_batch_size`: 32768
- `per_device_eval_batch_size`: 32768
- `learning_rate`: 3e-05
- `warmup_ratio`: 0.03333333333333333
- `fp16`: True
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32768
- `per_device_eval_batch_size`: 32768
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `learning_rate`: 3e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.03333333333333333
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
- `dataloader_drop_last`: True
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
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional

</details>


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
```bibtex
@misc{bge-m3,
      title={BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation}, 
      author={Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
      year={2024},
      eprint={2402.03216},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
```bibtex
@article{wang2022text,
  title={Text Embeddings by Weakly-Supervised Contrastive Pre-training},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Jiao, Binxing and Yang, Linjun and Jiang, Daxin and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2212.03533},
  year={2022}
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