# RAG in conjunction with LLaMA3 to align content classifiers (Handout) *Maxyn Leitner*

## Our Use Case: Labeling Trans Related Sentiment

## Our Taxonomy of Trans Related Sentiment

![](data:image/png;base64...)

## Our RAG Pipeline

![](data:image/png;base64...)

### Sample Codebook Sentences/RAG Database entries:

* Trans/nonbinary people affirming their own identities should be rated as **pro-transgender**, with the sublabel *Celebration of Trans Existence*.
* Any language claiming that there's only two genders, or that your gender is what you were assigned at birth, should be rated as **anti-transgender** with the sublabel *Exorsexism*.
* Examples claiming to protect women's sports from "men" (referring to trans women) should be labeled as **anti-transgender** with the sublabel *Transmisogyny*.

## Troubleshooting Tips

Dials to tweak:

* similarity cutoff
* similarity k

More substantial interventions:

* Add/remove database items
* Reword RAG sentences

Try these last:

* max iterations
* temperature

Bonus:

* In the annotation phase, can have annotators indicate which sentences inform their classification.
* This can then be used for Retrieval Augmented Fine Tuning (RAFT)

### Resources

#### Video of naive vs RAG informed Content Generation

<https://youtu.be/rDl6zrRtDyE>

#### Sample code

<https://github.com/maxynrl/KnitTogetherReference/blob/main/antitransRAG-KT.py>

#### Guide to getting a HuggingFace access token:

<https://huggingface.co/docs/hub/security-tokens>

#### More details on querying RAG Databases:

<https://docs.llamaindex.ai/en/v0.10.17/understanding/querying/querying.html>

#### Explainer on Similarity Matching with Threshold:

<https://meisinlee.medium.com/better-rag-retrieval-similarity-with-threshold-a6dbb535ef9e>
