<h1 align="center"> Natural Language Processing  with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : Kharis Hendroh Prasetyo Ogotan

## My todo : 

### 1. Example 1 - Sentiment Analysis

```
# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I am so happy that my code is running well, today is a good day")
```

Result : 

```
[{'label': 'POSITIVE', 'score': 0.9998620748519897}]
```

Analysis on example 1 : 

The model correctly predicts the sentiment as positive with a 99.9% confidence score. It highlights the classifier's high accuracy in detecting clear expressions of joy and satisfaction.


### 2. Example 2 - Topic Classification

```
# TODO :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "Building a seamless full-stack application using Laravel for the backend and Flutter for the mobile UI is incredibly rewarding.",
    candidate_labels=["technology", "sports", "cooking", "education"],
)
```

Result : 

```
{
  "sequence": "Building a seamless full-stack application using Laravel for the backend and Flutter for the mobile UI is incredibly rewarding.",
  "labels": ["technology", "cooking", "sports", "education"],
  "scores": [0.9714969396591187, 0.01880190335214138, 0.00493127154186368, 0.0047698733339682817]
}
```

Analysis on example 2 : 

The model successfully classifies the prompt under "technology" with a strong score of 97.1%. It accurately links tech framework keywords to the right category without prior domain training.

### 3. Example 3 and 3.5 - Text Generator

```
# TODO :
generator = pipeline("text-generation", model="gpt2")
generator(
    "In the near future, artificial intelligence and automated systems will completely transform how software developers write code",
    max_length=50,
    num_return_sequences=2
)
```

Result : 

```
[
  {"generated_text": "In the near future, artificial intelligence and automated systems will completely transform how software developers write code, and it\\'s the future that\\'s being explored by companies like Google.\\n\\n\\n\"We\\'re going to see these systems learn a lot from the experience of a human,\" said Ravi Kishore, Google\\'s director of AI and Systems Design. \"It\\'s not going to be a simple task, it\\'s going to be a long, complex process. We have to deliver on this.\""},
  {"generated_text": "In the near future, artificial intelligence and automated systems will completely transform how software developers write code. The potential for this transformation is enormous.\\n\\n\"In a world where computers are being developed at a rapid pace, we are seeing the ability to do more with less. That\\'s exciting. And it gives us a major advantage.\"\\n\\nThe fact that AI is so rapidly evolving and in many cases the most advanced form of machine intelligence is being developed at such a fast pace, adds to the potential for the future of software development.\\n\\nThe technology is evolving rapidly, and this is leading to a new level of competition between the different AI platforms.\\n\\nAnd AI is also being developed at a faster pace.\\n\\n\"It is really exciting and exciting to be an AI developer,\" says Richard Stallman, co-founder and CEO of AI startup Infusion. \"You are seeing a lot of things happening all at once.\\n\\n\"I am very excited about this. It is very exciting to be an AI developer and now we are making that possible.\"\\n\\nThe company is also aiming to grow its development and testing activities to be able to scale.\\n\\n\"We are not talking about making things for you, we are talking about'"}
]
```

Analysis on example 3 : 

The model generates coherent, context-aware completions based on the AI development prompt. It creates logical narratives about industry trends and engineering impacts while maintaining smooth text flow.

```
# TODO :
unmasker = pipeline("fill-mask", model="albert-base-v2")
unmasker("Every programmer needs to write clean [MASK] to make it easy to maintain.")
```

Result : 

```
[{'score': 0.6125652194023132,
  'token': 17505,
  'token_str': 'scripts',
  'sequence': 'every programmer needs to write clean scripts to make it easy to maintain.'},
 {'score': 0.034432828426361084,
  'token': 3884,
  'token_str': 'script',
  'sequence': 'every programmer needs to write clean script to make it easy to maintain.'},
 {'score': 0.032169975340366364,
  'token': 13453,
  'token_str': 'sentences',
  'sequence': 'every programmer needs to write clean sentences to make it easy to maintain.'},
 {'score': 0.024171296507120132,
  'token': 1797,
  'token_str': 'code',
  'sequence': 'every programmer needs to write clean code to make it easy to maintain.'},
 {'score': 0.013783860951662064,
  'token': 4374,
  'token_str': 'documents',
  'sequence': 'every programmer needs to write clean documents to make it easy to maintain.'}]
```

Analysis on example 3.5 : 

The fill-mask model accurately predicts programming-related terms to complete the sentence. It ranks "scripts" as the top option with 61.2% confidence, demonstrating strong contextual understanding of software development terms.

### 4. Example 4 - Name Entity Recognition (NER)

```
# TODO :
ner = pipeline("ner", model="dslim/bert-large-NER", aggregation_strategy="simple")
ner("Kharis and his tech team are building a web application using Laravel at Universitas Indonesia.")
```

Result : 

```
[{'entity_group': 'PER',
  'score': np.float32(0.9406583),
  'word': 'Kharis',
  'start': 0,
  'end': 6},
 {'entity_group': 'MISC',
  'score': np.float32(0.7945755),
  'word': 'Laravel',
  'start': 62,
  'end': 69},
 {'entity_group': 'ORG',
  'score': np.float32(0.9982846),
  'word': 'Universitas Indonesia',
  'start': 73,
  'end': 94}]
```

Analysis on example 4 : 

The model demonstrates high technical precision in entity extraction. It correctly maps "Kharis" as a person (PER), "Laravel" as a technical project/framework name (MISC), and "Universitas Indonesia" as an organization (ORG).

### 5. Example 5 - Question Answering

```
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What is the name of the application Kharis is developing?"
context = "Kharis and his capstone project group are developing an innovative application called Home Cycle, which uses artificial intelligence to manage food waste."
qa_model(question = question, context = context)
```

Result : 

```
{'score': 0.996967613697052, 'start': 86, 'end': 96, 'answer': 'Home Cycle'}
```

Analysis on example 5 : 

The model isolates and extracts the exact target answer string "Home Cycle" from the text block with an exceptionally high score of 99.6%, proving its strong performance in factual informational lookups.

### 6. Example 6 - Text Summarization

```
# TODO :
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text_panjang = """
Artificial intelligence is rapidly evolving and playing a crucial role in modern software engineering. 
Many developers are now integrating machine learning models into web and mobile applications to provide smarter user experiences. 
For example, automated systems can now predict user behavior, optimize cloud infrastructure, and even detect bugs in the code before deployment. 
As a result, understanding how to work with pre-trained models from platforms like Hugging Face has become an essential skill for the next generation of IT students.
"""

inputs = tokenizer([text_panjang], max_length=1024, return_tensors="pt", truncation=True)

summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=35, min_length=10, early_stopping=True)
summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print([{"summary_text": summary_text}])
```

Result : 

```
[{'summary_text': ' Artificial intelligence is rapidly evolving and playing a crucial role in software engineering . Understanding how to work with pre-trained models from platforms like Hugging Face has become an'}]

```

Analysis on example 6 :

Using the direct manual decoding workaround via AutoModelForSeq2SeqLM, the model successfully condenses the paragraph. It extracts the essential geographic and demographic metrics of the text while dropping secondary descriptors.

### 7. Example 7 - Translation

```
# TODO :
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "Helsinki-NLP/opus-mt-en-id"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

kalimat_inggris = "Learning data science and artificial intelligence will open up many brilliant career opportunities in the future."

inputs = tokenizer([kalimat_inggris], return_tensors="pt", padding=True)

translated_ids = model.generate(**inputs)
translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

print([{"translation_text": translated_text}])
```

Result : 

```
[{'translation_text': 'Belajar ilmu data dan kecerdasan buatan akan membuka banyak peluang karir brilian di masa depan.'}]

```

Analysis on example 7 :

The sequence-to-sequence translation architecture performs exceptionally well, outputting a highly accurate and natural Indonesian conversion. The model handles context phrasing correctly, delivering smooth and accurate language mapping.

---

## Analysis on this project

Through this project, practical implementations of Hugging Face Transformers are demonstrated across various NLP domains. The real-world scenarios emphasize how flexible and powerful pre-trained models can be when addressing different text-based problem sets.
