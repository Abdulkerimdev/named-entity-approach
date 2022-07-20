```python
# Load a spacy model and chekc if it has ner
import spacy
from spacy import displacy

nlp=spacy.load('en_core_web_trf')


```

    C:\Users\abdul\AppData\Local\Programs\Python\Python39\lib\site-packages\tqdm\auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    


```python

```


```python
article_text = """ 
Ezio Auditore da Firenze (1459–1524) was a Florentine nobleman during the Renaissance, and, unbeknownst to most historians and philosophers, the Mentor of the Italian Brotherhood of Assassins, a title which he held from 1503 to 1513. He is also an ancestor to William and Desmond Miles, as well as Clay Kaczmarek. A member of the House of Auditore, Ezio remained unaware of his Assassin heritage until the age of 17, when he witnessed the hanging of his father and two brothers, Federico and Petruccio. Forced to flee his birthplace with his remaining family members - his mother and sister - Ezio took refuge in the Tuscan town of Monteriggioni, at the Villa Auditore. After learning of his heritage from his uncle, Mario Auditore, Ezio began his Assassin training and set about on his quest for vengeance against the Templar Order, and their Grand Master, Rodrigo Borgia, who had ordered the execution of his kin. During his travels, Ezio managed to not only unite the pages of the Codex, written by Altaïr Ibn-La'Ahad, Mentor of the Levantine Assassins, but also to save the cities of Florence, Venice, and Rome from Templar rule."""
```


```python
doc=nlp(article_text)
for ent in doc.ents:
  print(ent.text,ent.label_)


```

    C:\Users\abdul\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\amp\autocast_mode.py:198: UserWarning: User provided device_type of 'cuda', but CUDA is not available. Disabling
      warnings.warn('User provided device_type of \'cuda\', but CUDA is not available. Disabling')
    

    Ezio Auditore da Firenze PERSON
    Florentine NORP
    the Italian Brotherhood of Assassins ORG
    from 1503 to 1513 DATE
    William PERSON
    Desmond Miles PERSON
    Clay Kaczmarek PERSON
    Ezio PERSON
    Assassin NORP
    the age of 17 DATE
    two CARDINAL
    Federico PERSON
    Petruccio PERSON
    Ezio PERSON
    Tuscan NORP
    Monteriggioni GPE
    the Villa Auditore FAC
    Mario Auditore PERSON
    Ezio PERSON
    Assassin ORG
    the Templar Order ORG
    Rodrigo Borgia PERSON
    Ezio PERSON
    the Codex WORK_OF_ART
    Altaïr Ibn-La'Ahad PERSON
    Levantine NORP
    Florence GPE
    Venice GPE
    Rome GPE
    Templar ORG
    


```python
displacy.render(doc, style="ent", jupyter = True)

```


<span class="tex2jax_ignore"><div class="entities" style="line-height: 2.5; direction: ltr"> </br>
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Ezio Auditore da Firenze
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 (1459–1524) was a 
<mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Florentine
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">NORP</span>
</mark>
 nobleman during the Renaissance, and, unbeknownst to most historians and philosophers, the Mentor of 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the Italian Brotherhood of Assassins
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
, a title which he held 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    from 1503 to 1513
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
. He is also an ancestor to 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    William
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 and 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Desmond Miles
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
, as well as 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Clay Kaczmarek
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. A member of the House of Auditore, 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Ezio
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 remained unaware of his 
<mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Assassin
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">NORP</span>
</mark>
 heritage until 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the age of 17
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
, when he witnessed the hanging of his father and 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    two
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 brothers, 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Federico
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 and 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Petruccio
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
. Forced to flee his birthplace with his remaining family members - his mother and sister - 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Ezio
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 took refuge in the 
<mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Tuscan
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">NORP</span>
</mark>
 town of 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Monteriggioni
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, at 
<mark class="entity" style="background: #9cc9cc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the Villa Auditore
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">FAC</span>
</mark>
. After learning of his heritage from his uncle, 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Mario Auditore
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
, 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Ezio
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 began his 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Assassin
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 training and set about on his quest for vengeance against 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the Templar Order
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
, and their Grand Master, 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Rodrigo Borgia
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
, who had ordered the execution of his kin. During his travels, 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Ezio
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 managed to not only unite the pages of 
<mark class="entity" style="background: #f0d0ff; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the Codex
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">WORK_OF_ART</span>
</mark>
, written by 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Altaïr Ibn-La'Ahad
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
, Mentor of the 
<mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Levantine
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">NORP</span>
</mark>
 Assassins, but also to save the cities of 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Florence
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Venice
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, and 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Rome
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 from 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Templar
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 rule.</div></span>



```python

TRAIN_DATA = [
    ('Who is Ezio?', {
        'entities': [(7, 10, 'PERSON')]
    }),
     ('Who is Rodrigo Borgia?', {
        'entities': [(7, 20, 'PERSON')]
    }),
    ('I like William Miles and Desmond Miles.', {
        'entities': [(7, 19, 'PERSON'), (25, 37, 'PERSON')]
    }),
     ('I born in Tuscan.', {
        'entities': [(7, 12, 'LOC')]
    }),
    ('Pages of Codex are missing' , {
        'entities': [(9, 13, 'WORK_OF_ART')]
    }),
     ('managed to not only unite the pages of the Codex, written by Altaïr Ibn-LaAhad' , {
        'entities': [(43, 47, 'WORK_OF_ART') ,(61,77, 'PERSON')]
    })
]
 
```


```python
doc2=nlp("He is also an ancestor to William and Desmond Miles, as well as Clay Kaczmarek.")

```


```python
svg1=displacy.render(doc2, style="dep")

```


<span class="tex2jax_ignore"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="en" id="8c35b35f63c54db5be7a6648f9cf7678-0" class="displacy" width="2675" height="487.0" direction="ltr" style="max-width: none; height: 487.0px; color: #000000; background: #ffffff; font-family: Arial; direction: ltr">
<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="50">He</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="50">PRON</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="225">is</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="225">AUX</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="400">also</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="400">ADV</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="575">an</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="575">DET</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="750">ancestor</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="750">NOUN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="925">to</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="925">ADP</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="1100">William</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1100">PROPN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="1275">and</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1275">CCONJ</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="1450">Desmond</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1450">PROPN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="1625">Miles,</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1625">PROPN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="1800">as</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1800">ADV</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="1975">well</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1975">ADV</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="2150">as</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="2150">ADP</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="2325">Clay</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="2325">PROPN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="2500">Kaczmarek.</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="2500">PROPN</tspan>
</text>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-8c35b35f63c54db5be7a6648f9cf7678-0-0" stroke-width="2px" d="M70,352.0 C70,264.5 210.0,264.5 210.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-8c35b35f63c54db5be7a6648f9cf7678-0-0" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">nsubj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M70,354.0 L62,342.0 78,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-8c35b35f63c54db5be7a6648f9cf7678-0-1" stroke-width="2px" d="M245,352.0 C245,264.5 385.0,264.5 385.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-8c35b35f63c54db5be7a6648f9cf7678-0-1" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">advmod</textPath>
    </text>
    <path class="displacy-arrowhead" d="M385.0,354.0 L393.0,342.0 377.0,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-8c35b35f63c54db5be7a6648f9cf7678-0-2" stroke-width="2px" d="M595,352.0 C595,264.5 735.0,264.5 735.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-8c35b35f63c54db5be7a6648f9cf7678-0-2" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">det</textPath>
    </text>
    <path class="displacy-arrowhead" d="M595,354.0 L587,342.0 603,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-8c35b35f63c54db5be7a6648f9cf7678-0-3" stroke-width="2px" d="M245,352.0 C245,177.0 740.0,177.0 740.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-8c35b35f63c54db5be7a6648f9cf7678-0-3" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">attr</textPath>
    </text>
    <path class="displacy-arrowhead" d="M740.0,354.0 L748.0,342.0 732.0,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-8c35b35f63c54db5be7a6648f9cf7678-0-4" stroke-width="2px" d="M770,352.0 C770,264.5 910.0,264.5 910.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-8c35b35f63c54db5be7a6648f9cf7678-0-4" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">prep</textPath>
    </text>
    <path class="displacy-arrowhead" d="M910.0,354.0 L918.0,342.0 902.0,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-8c35b35f63c54db5be7a6648f9cf7678-0-5" stroke-width="2px" d="M1120,352.0 C1120,89.5 1620.0,89.5 1620.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-8c35b35f63c54db5be7a6648f9cf7678-0-5" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">nmod</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1120,354.0 L1112,342.0 1128,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-8c35b35f63c54db5be7a6648f9cf7678-0-6" stroke-width="2px" d="M1120,352.0 C1120,264.5 1260.0,264.5 1260.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-8c35b35f63c54db5be7a6648f9cf7678-0-6" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">cc</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1260.0,354.0 L1268.0,342.0 1252.0,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-8c35b35f63c54db5be7a6648f9cf7678-0-7" stroke-width="2px" d="M1120,352.0 C1120,177.0 1440.0,177.0 1440.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-8c35b35f63c54db5be7a6648f9cf7678-0-7" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">conj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1440.0,354.0 L1448.0,342.0 1432.0,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-8c35b35f63c54db5be7a6648f9cf7678-0-8" stroke-width="2px" d="M945,352.0 C945,2.0 1625.0,2.0 1625.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-8c35b35f63c54db5be7a6648f9cf7678-0-8" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">pobj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1625.0,354.0 L1633.0,342.0 1617.0,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-8c35b35f63c54db5be7a6648f9cf7678-0-9" stroke-width="2px" d="M1820,352.0 C1820,177.0 2140.0,177.0 2140.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-8c35b35f63c54db5be7a6648f9cf7678-0-9" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">advmod</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1820,354.0 L1812,342.0 1828,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-8c35b35f63c54db5be7a6648f9cf7678-0-10" stroke-width="2px" d="M1995,352.0 C1995,264.5 2135.0,264.5 2135.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-8c35b35f63c54db5be7a6648f9cf7678-0-10" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">advmod</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1995,354.0 L1987,342.0 2003,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-8c35b35f63c54db5be7a6648f9cf7678-0-11" stroke-width="2px" d="M1645,352.0 C1645,89.5 2145.0,89.5 2145.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-8c35b35f63c54db5be7a6648f9cf7678-0-11" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">cc</textPath>
    </text>
    <path class="displacy-arrowhead" d="M2145.0,354.0 L2153.0,342.0 2137.0,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-8c35b35f63c54db5be7a6648f9cf7678-0-12" stroke-width="2px" d="M2345,352.0 C2345,264.5 2485.0,264.5 2485.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-8c35b35f63c54db5be7a6648f9cf7678-0-12" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">compound</textPath>
    </text>
    <path class="displacy-arrowhead" d="M2345,354.0 L2337,342.0 2353,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-8c35b35f63c54db5be7a6648f9cf7678-0-13" stroke-width="2px" d="M1645,352.0 C1645,2.0 2500.0,2.0 2500.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-8c35b35f63c54db5be7a6648f9cf7678-0-13" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">conj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M2500.0,354.0 L2508.0,342.0 2492.0,342.0" fill="currentColor"/>
</g>
</svg></span>



```python
ner=nlp.get_pipe('ner')


```


```python

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

nlp = spacy.blank("en") # load a new spacy model
doc_bin = DocBin() # create a DocBin object
```


```python
import json
import os
notebook_path = os.path.abspath("ai-odev.ipynb")
file=os.path.join(os.path.dirname(notebook_path), "data.json")
 
with open(file, 'r') as f:
    data = json.load(f)
    
print(data['examples'][0])
```

    {'id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'content': "While bismuth compounds (Pepto-Bismol) decreased the number of bowel movements in those with travelers' diarrhea, they do not decrease the length of illness.[91] Anti-motility agents like loperamide are also effective at reducing the number of stools but not the duration of disease.[8] These agents should be used only if bloody diarrhea is not present.[92]\n\nDiosmectite, a natural aluminomagnesium silicate clay, is effective in alleviating symptoms of acute diarrhea in children,[93] and also has some effects in chronic functional diarrhea, radiation-induced diarrhea, and chemotherapy-induced diarrhea.[45] Another absorbent agent used for the treatment of mild diarrhea is kaopectate.\n\nRacecadotril an antisecretory medication may be used to treat diarrhea in children and adults.[86] It has better tolerability than loperamide, as it causes less constipation and flatulence.[94]", 'metadata': {}, 'annotations': [{'id': '0825a1bf-6a6e-4fa2-be77-8d104701eaed', 'tag_id': 'c06bd022-6ded-44a5-8d90-f17685bb85a1', 'end': 371, 'start': 360, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'Medicine', 'value': 'Diosmectite', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:24:32.098000Z', 'annotator_id': 1, 'tagged_token_id': '0825a1bf-6a6e-4fa2-be77-8d104701eaed', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}, {'id': '145f62b1-bbf5-42f1-8ad5-9c7e08337bf0', 'tag_id': 'c06bd022-6ded-44a5-8d90-f17685bb85a1', 'end': 408, 'start': 383, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'Medicine', 'value': 'aluminomagnesium silicate', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:24:43.692000Z', 'annotator_id': 1, 'tagged_token_id': '145f62b1-bbf5-42f1-8ad5-9c7e08337bf0', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}, {'id': '243efeb2-723f-4be4-933c-fbbf7e0f9903', 'tag_id': '03eb3e50-d4d8-4261-a60b-fa5aee5deb4a', 'end': 112, 'start': 104, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'MedicalCondition', 'value': 'diarrhea', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:24:09.423000Z', 'annotator_id': 1, 'tagged_token_id': '243efeb2-723f-4be4-933c-fbbf7e0f9903', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}, {'id': '281f49d3-879a-4409-b09e-4cfae019af16', 'tag_id': 'c06bd022-6ded-44a5-8d90-f17685bb85a1', 'end': 689, 'start': 679, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'Medicine', 'value': 'kaopectate', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:25:38.366000Z', 'annotator_id': 1, 'tagged_token_id': '281f49d3-879a-4409-b09e-4cfae019af16', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}, {'id': '32fdf9e7-63f7-442a-8e25-f08ea4ad94d5', 'tag_id': 'c06bd022-6ded-44a5-8d90-f17685bb85a1', 'end': 23, 'start': 6, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'Medicine', 'value': 'bismuth compounds', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:23:46.721000Z', 'annotator_id': 1, 'tagged_token_id': '32fdf9e7-63f7-442a-8e25-f08ea4ad94d5', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}, {'id': '392094d2-febf-4074-a2ca-4c0082f4e5b8', 'tag_id': 'c06bd022-6ded-44a5-8d90-f17685bb85a1', 'end': 37, 'start': 25, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'Medicine', 'value': 'Pepto-Bismol', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:23:58.861000Z', 'annotator_id': 1, 'tagged_token_id': '392094d2-febf-4074-a2ca-4c0082f4e5b8', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}, {'id': '450b8c30-cf2e-4774-96d9-58b4160bea38', 'tag_id': '03eb3e50-d4d8-4261-a60b-fa5aee5deb4a', 'end': 470, 'start': 461, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'MedicalCondition', 'value': 'diarrhea ', 'correct': None, 'human_annotations': [], 'model_annotations': []}, {'id': '6b73f683-7130-4e16-bcc2-e3cc8cf89f4d', 'tag_id': 'c06bd022-6ded-44a5-8d90-f17685bb85a1', 'end': 589, 'start': 577, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'Medicine', 'value': 'chemotherapy', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:25:24.179000Z', 'annotator_id': 1, 'tagged_token_id': '6b73f683-7130-4e16-bcc2-e3cc8cf89f4d', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}, {'id': '74574b7f-d535-48e1-8651-2708adcfe453', 'tag_id': '03eb3e50-d4d8-4261-a60b-fa5aee5deb4a', 'end': 865, 'start': 853, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'MedicalCondition', 'value': 'constipation', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:26:31.676000Z', 'annotator_id': 1, 'tagged_token_id': '74574b7f-d535-48e1-8651-2708adcfe453', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}, {'id': '7572ab8e-ae99-400c-b4ab-ed46fbc9f97e', 'tag_id': 'c06bd022-6ded-44a5-8d90-f17685bb85a1', 'end': 198, 'start': 188, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'Medicine', 'value': 'loperamide', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:24:17.680000Z', 'annotator_id': 1, 'tagged_token_id': '7572ab8e-ae99-400c-b4ab-ed46fbc9f97e', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}, {'id': '800e6c6c-0bfb-4819-a25a-34f759753457', 'tag_id': '03eb3e50-d4d8-4261-a60b-fa5aee5deb4a', 'end': 762, 'start': 754, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'MedicalCondition', 'value': 'diarrhea', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:26:07.368000Z', 'annotator_id': 1, 'tagged_token_id': '800e6c6c-0bfb-4819-a25a-34f759753457', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}, {'id': '8214556a-7584-4d9b-86cd-5e09137ad904', 'tag_id': '03eb3e50-d4d8-4261-a60b-fa5aee5deb4a', 'end': 880, 'start': 870, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'MedicalCondition', 'value': 'flatulence', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:26:41.322000Z', 'annotator_id': 1, 'tagged_token_id': '8214556a-7584-4d9b-86cd-5e09137ad904', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}, {'id': '98968e14-6756-4174-9d3d-7abd58b3aa34', 'tag_id': 'c06bd022-6ded-44a5-8d90-f17685bb85a1', 'end': 833, 'start': 823, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'Medicine', 'value': 'loperamide', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:26:12.759000Z', 'annotator_id': 1, 'tagged_token_id': '98968e14-6756-4174-9d3d-7abd58b3aa34', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}, {'id': 'a0a03c7b-cfad-41ee-9f5c-f8a802475994', 'tag_id': '03eb3e50-d4d8-4261-a60b-fa5aee5deb4a', 'end': 853, 'start': 852, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'MedicalCondition', 'value': ' ', 'correct': None, 'human_annotations': [], 'model_annotations': []}, {'id': 'bfbddfd4-38aa-43a7-9366-24c95829ac8c', 'tag_id': '03eb3e50-d4d8-4261-a60b-fa5aee5deb4a', 'end': 469, 'start': 461, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'MedicalCondition', 'value': 'diarrhea', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:25:06.921000Z', 'annotator_id': 1, 'tagged_token_id': 'bfbddfd4-38aa-43a7-9366-24c95829ac8c', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}, {'id': 'd7d68c18-0d8e-4547-a2fa-81fdcaf3080e', 'tag_id': '03eb3e50-d4d8-4261-a60b-fa5aee5deb4a', 'end': 543, 'start': 535, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'MedicalCondition', 'value': 'diarrhea', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:25:12.203000Z', 'annotator_id': 1, 'tagged_token_id': 'd7d68c18-0d8e-4547-a2fa-81fdcaf3080e', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}, {'id': 'ee956220-42a4-4a91-b9f0-75019c4f5dd9', 'tag_id': 'c06bd022-6ded-44a5-8d90-f17685bb85a1', 'end': 704, 'start': 692, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'Medicine', 'value': 'Racecadotril', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:25:56.503000Z', 'annotator_id': 1, 'tagged_token_id': 'ee956220-42a4-4a91-b9f0-75019c4f5dd9', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}, {'id': 'f04a6b7e-8904-405c-9301-e4543238b7a5', 'tag_id': '03eb3e50-d4d8-4261-a60b-fa5aee5deb4a', 'end': 571, 'start': 563, 'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe', 'tag_name': 'MedicalCondition', 'value': 'diarrhea', 'correct': None, 'human_annotations': [{'timestamp': '2020-03-21T00:25:18.043000Z', 'annotator_id': 1, 'tagged_token_id': 'f04a6b7e-8904-405c-9301-e4543238b7a5', 'name': 'Ashpat123', 'reason': 'exploration'}], 'model_annotations': []}], 'classifications': []}
    


```python
training_data = {'classes' : ['MEDICINE', "MEDICALCONDITION", "PATHOGEN"], 'annotations' : []}
for example in data['examples']:
  temp_dict = {}
  temp_dict['text'] = example['content']
  temp_dict['entities'] = []
  for annotation in example['annotations']:
    start = annotation['start']
    end = annotation['end']
    label = annotation['tag_name'].upper()
    temp_dict['entities'].append((start, end, label))
  training_data['annotations'].append(temp_dict)
  
print(training_data['annotations'][0])
```

    {'text': "While bismuth compounds (Pepto-Bismol) decreased the number of bowel movements in those with travelers' diarrhea, they do not decrease the length of illness.[91] Anti-motility agents like loperamide are also effective at reducing the number of stools but not the duration of disease.[8] These agents should be used only if bloody diarrhea is not present.[92]\n\nDiosmectite, a natural aluminomagnesium silicate clay, is effective in alleviating symptoms of acute diarrhea in children,[93] and also has some effects in chronic functional diarrhea, radiation-induced diarrhea, and chemotherapy-induced diarrhea.[45] Another absorbent agent used for the treatment of mild diarrhea is kaopectate.\n\nRacecadotril an antisecretory medication may be used to treat diarrhea in children and adults.[86] It has better tolerability than loperamide, as it causes less constipation and flatulence.[94]", 'entities': [(360, 371, 'MEDICINE'), (383, 408, 'MEDICINE'), (104, 112, 'MEDICALCONDITION'), (679, 689, 'MEDICINE'), (6, 23, 'MEDICINE'), (25, 37, 'MEDICINE'), (461, 470, 'MEDICALCONDITION'), (577, 589, 'MEDICINE'), (853, 865, 'MEDICALCONDITION'), (188, 198, 'MEDICINE'), (754, 762, 'MEDICALCONDITION'), (870, 880, 'MEDICALCONDITION'), (823, 833, 'MEDICINE'), (852, 853, 'MEDICALCONDITION'), (461, 469, 'MEDICALCONDITION'), (535, 543, 'MEDICALCONDITION'), (692, 704, 'MEDICINE'), (563, 571, 'MEDICALCONDITION')]}
    


```python
from spacy.tokens import DocBin
from tqdm import tqdm

nlp = spacy.blank("en") # load a new spacy model
doc_bin = DocBin() # create a DocBin object
```


```python

from spacy.util import filter_spans

for training_example  in tqdm(training_data['annotations']): 
    text = training_example['text']
    labels = training_example['entities']
    doc = nlp.make_doc(text) 
    ents = []
    for start, end, label in labels:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    filtered_ents = filter_spans(ents)
    doc.ents = filtered_ents 
    doc_bin.add(doc)

doc_bin.to_disk("training_data.spacy") # save the docbin object
```

    100%|█████████████████████████████████████████████████████████████████████████████████| 31/31 [00:00<00:00, 612.91it/s]

    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    Skipping entity
    

    
    


```python


```


```python
!python -m spacy train config.cfg --output ./ --paths.train ./training_data.spacy --paths.dev ./training_data.spacy 

```

    [i] Saving to output directory: .
    [i] Using CPU
    [1m
    =========================== Initializing pipeline ===========================[0m
    [+] Initialized pipeline
    [1m
    ============================= Training pipeline =============================[0m
    [i] Pipeline: ['tok2vec', 'ner']
    [i] Initial learn rate: 0.001
    E    #       LOSS TOK2VEC  LOSS NER  ENTS_F  ENTS_P  ENTS_R  SCORE 
    ---  ------  ------------  --------  ------  ------  ------  ------
      0       0          0.00    150.79    1.96    1.90    2.03    0.02
      7     200        625.16   3256.64   73.02   71.32   74.80    0.73
     14     400        168.82    635.78   95.69   96.68   94.72    0.96
     22     600        206.33    256.97   96.72   97.52   95.93    0.97
     30     800        383.08    249.60   97.36   97.17   97.56    0.97
     40    1000        201.78    188.60   97.98   97.58   98.37    0.98
     52    1200        208.84    174.71   98.37   98.37   98.37    0.98
     65    1400        160.35    152.01   98.37   98.37   98.37    0.98
     82    1600        380.60    196.42   98.38   97.98   98.78    0.98
    103    1800        210.74    220.21   98.37   98.37   98.37    0.98
    128    2000        146.74    222.88   98.37   98.37   98.37    0.98
    160    2200        204.24    269.74   98.37   98.37   98.37    0.98
    200    2400        205.97    290.93   98.79   98.39   99.19    0.99
    249    2600        173.04    283.50   98.78   98.78   98.78    0.99
    298    2800        144.89    263.47   98.79   98.39   99.19    0.99
    348    3000        101.98    290.01   98.78   99.18   98.37    0.99
    397    3200         79.82    257.77   98.79   98.39   99.19    0.99
    447    3400         87.90    257.95   98.79   98.39   99.19    0.99
    496    3600        177.55    274.56   98.79   98.39   99.19    0.99
    545    3800        153.66    254.39   98.78   98.78   98.78    0.99
    595    4000         82.75    239.79   98.79   98.39   99.19    0.99
    [+] Saved pipeline to output directory
    model-last
    

    [2022-07-20 15:28:21,614] [INFO] Set up nlp object from config
    [2022-07-20 15:28:21,622] [INFO] Pipeline: ['tok2vec', 'ner']
    [2022-07-20 15:28:21,624] [INFO] Created vocabulary
    [2022-07-20 15:28:21,625] [INFO] Finished initializing nlp object
    [2022-07-20 15:28:21,872] [INFO] Initialized pipeline components: ['tok2vec', 'ner']
    


```python
nlp_ner = spacy.load("model-best")

doc = nlp_ner("Antiretroviral therapy (ART) is recommended for all HIV-infected\
individuals to reduce the risk of disease progression.\nART also is recommended \
for HIV-infected individuals for the prevention of transmission of HIV.\nPatients \
starting ART should be willing and able to commit to treatment and understand the\
benefits and risks of therapy and the importance of adherence. Patients may choose\
to postpone therapy, and providers, on a case-by-case basis, may elect to defer\
therapy on the basis of clinical and/or psychosocial factors.")

colors = {"PATHOGEN": "#F67DE3", "MEDICINE": "#7DF6D9", "MEDICALCONDITION":"#FFFFFF"}
options = {"colors": colors} 

spacy.displacy.render(doc, style="ent", options= options, jupyter=True)
```


<span class="tex2jax_ignore"><div class="entities" style="line-height: 2.5; direction: ltr">
<mark class="entity" style="background: #7DF6D9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Antiretroviral therapy
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MEDICINE</span>
</mark>
 (
<mark class="entity" style="background: #7DF6D9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    ART
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MEDICINE</span>
</mark>
) is recommended for all 
<mark class="entity" style="background: #F67DE3; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    HIV
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PATHOGEN</span>
</mark>
-infectedindividuals to reduce the risk of disease progression.</br>
<mark class="entity" style="background: #7DF6D9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    ART
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MEDICINE</span>
</mark>
 also is recommended for 
<mark class="entity" style="background: #F67DE3; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    HIV
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PATHOGEN</span>
</mark>
-infected individuals for the prevention of transmission of 
<mark class="entity" style="background: #F67DE3; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    HIV
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PATHOGEN</span>
</mark>
.</br>Patients starting 
<mark class="entity" style="background: #7DF6D9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    ART
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MEDICINE</span>
</mark>
 should be willing and able to commit to treatment and understand thebenefits and risks of therapy and the importance of adherence. Patients may 
<mark class="entity" style="background: #FFFFFF; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    chooseto postpone therapy
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MEDICALCONDITION</span>
</mark>
, and providers, on a case-by-case basis, may elect to defertherapy on the basis of clinical and/or psychosocial factors.</div></span>



```python
nlp_ner2 = spacy.load("model-best")

doc2 = nlp_ner2("Antiretroviral therapy (ART) is recommended for all HIV-infected\
individuals to reduce the risk of disease progression.\nART also is recommended \
for HIV-infected individuals for the prevention of transmission of HIV.\nPatients \
starting ART should be willing and able to commit to treatment and understand the\
benefits and risks of therapy and the importance of adherence. Patients may choose\
to postpone therapy, and providers, on a case-by-case basis, may elect to defer\
therapy on the basis of clinical and/or psychosocial factors.")

colors = {"PATHOGEN": "#F67DE3", "MEDICINE": "#7DF6D9", "MEDICALCONDITION":"#FFFFFF"}
options = {"colors": colors} 

spacy.displacy.render(doc2, style="ent", options= options, jupyter=True)
```


<span class="tex2jax_ignore"><div class="entities" style="line-height: 2.5; direction: ltr">
<mark class="entity" style="background: #7DF6D9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Antiretroviral therapy
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MEDICINE</span>
</mark>
 (
<mark class="entity" style="background: #7DF6D9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    ART
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MEDICINE</span>
</mark>
) is recommended for all 
<mark class="entity" style="background: #F67DE3; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    HIV
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PATHOGEN</span>
</mark>
-infectedindividuals to reduce the risk of disease progression.</br>
<mark class="entity" style="background: #7DF6D9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    ART
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MEDICINE</span>
</mark>
 also is recommended for 
<mark class="entity" style="background: #F67DE3; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    HIV
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PATHOGEN</span>
</mark>
-infected individuals for the prevention of transmission of 
<mark class="entity" style="background: #F67DE3; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    HIV
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PATHOGEN</span>
</mark>
.</br>Patients starting 
<mark class="entity" style="background: #7DF6D9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    ART
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MEDICINE</span>
</mark>
 should be willing and able to commit to treatment and understand thebenefits and risks of therapy and the importance of adherence. Patients may 
<mark class="entity" style="background: #FFFFFF; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    chooseto postpone therapy
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MEDICALCONDITION</span>
</mark>
, and providers, on a case-by-case basis, may elect to defertherapy on the basis of clinical and/or psychosocial factors.</div></span>



```python
nlp_ner3 = spacy.load("model-best")
colors = {"PATHOGEN": "#F67DE3", "MEDICALCONDITION":"#7DF6D9"}
options = {"colors": colors}
doc3 = nlp_ner3("""
Increasing blood flow towards the skin allows more heat to be lost to the environment. Sweat cools down the body when it evaporates off the skin. Dilating blood vessels results in lower blood pressure, which makes the heart work harder to push blood around the body. For people with pre-existing heart conditions, this can lead to heart attacks. Excess sweat can lead to the loss of salt from the body. In extreme cases, a low level of sodium in the blood can result in nausea and headaches. What causes people to die in heatwaves? During heatwaves, there is a higher incidence of death from cardiovascular and respiratory diseases.""")
spacy.displacy.render(doc3, style="ent", options= options, jupyter=True)
```


<span class="tex2jax_ignore"><div class="entities" style="line-height: 2.5; direction: ltr"></br>
<mark class="entity" style="background: #F67DE3; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Increasing blood
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PATHOGEN</span>
</mark>
 flow towards the skin allows more heat to be lost to the environment. Sweat cools down the body when it evaporates off the skin. Dilating blood vessels results in lower blood pressure, which makes the heart work harder to push blood around the body. For people with pre-existing heart conditions, this can 
<mark class="entity" style="background: #7DF6D9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    lead to heart attacks
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MEDICALCONDITION</span>
</mark>
. Excess sweat can lead to the loss of salt from the body. In extreme cases, a low level of sodium in the blood can result in nausea and headaches. What causes people to die in heatwaves? 
<mark class="entity" style="background: #F67DE3; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    During heatwaves
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PATHOGEN</span>
</mark>
, there is a higher incidence of 
<mark class="entity" style="background: #7DF6D9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    death
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MEDICALCONDITION</span>
</mark>
 from cardiovascular and respiratory diseases.</div></span>

