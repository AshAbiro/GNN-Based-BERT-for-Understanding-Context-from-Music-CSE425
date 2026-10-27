---
license: mit
language:
- en
tags:
- art
- music
- text-to-music
- music-to-text
pretty_name: LP-MusicCaps-MTT
size_categories:
- 10K<n<100K
---

======================================

**!important**: Be careful when using `caption_attribute_prediction` (We don't recommend to use)!

======================================

# Dataset Card for LP-MusicCaps-MTT

## Dataset Description

- **Repository:** [LP-MusicCaps repository](https://github.com/seungheondoh/lp-music-caps)
- **Paper:** [ArXiv](https://arxiv.org/abs/2307.16372)

## Dataset Summary

**LP-MusicCaps** is a Large Language Model based Pseudo Music Caption dataset for `text-to-music` and `music-to-text` tasks. We construct the music-to-caption pairs with tag-to-caption generation (using three existing multi-label tag datasets and four task instructions). The data sources are MusicCaps, Magnatagtune, and Million Song Dataset ECALS subset.

- **LP-MusicCaps MTT (This Repo)**: 22k Audio with 88k Caption. We utilize 188 unique tags in the [Magnatagtune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) to perform tag-to-caption generation through LLM. Magnatagtune consists of 26k music clips from 5,223 unique songs including genre, instrument, vocal, mood, perceptual tempo, origin, and sonority features. We used the full 188 tag vocabulary and did not generate captions for tracks that do not have associated tags (decreased to 22k).
- [LP-MusicCaps MSD](https://huggingface.co/datasets/seungheondoh/LP-MusicCaps-MSD): 0.5M Audio with 2.2M Caption
- [LP-MusicCaps MC](https://huggingface.co/datasets/seungheondoh/LP-MusicCaps-MC): 6k Audio with 22k Caption.


## Data Instances

Each instance in LP-MusicCaps MTT (This Repo) represents multiple image-text pair information with meta-attributes:

```
{
  'track_id': '1541',
  'title': 'Eyes Closed (The Seldon Plan)',
  'artist_name': 'Magnatune.com',
  'release': 'Magnatune At The CC Salon',
  'tag_top50': ['guitar', 'country', 'male', 'singing'],
  'tag_top188': ['guitar',
    'male singer',
    'country',
    'male vocals',
    'male',
    'singing'
  ],
  'caption_writing': 'This country song features twangy guitar riffs and heartfelt male vocals, with a male singer singing about love and loss.',
  'caption_summary': 'A male singer with a country style voice accompanies his guitar while singing.',
  'caption_paraphrase': 'This male artist croons in a deep, soulful voice over the twangy sounds of his guitar, crafting a classic country tune perfect for fans of male vocals and raw, authentic singing.',
  'caption_attribute_prediction': 'A twangy mix of acoustic guitar and male vocals come together in this heartfelt country song. With lyrics that evoke a sense of nostalgia, the male singer weaves a story of love and loss through his storytelling. His emotive singing grips you from start to finish, as he sings about the trials and tribulations of life. This song is a must-listen for any fan of country.',
  'pseudo_attribute': ['acoustic',
    'twangy',
    'heartfelt',
    'storytelling',
    'nostalgic'
  ],
  'path': 'e/magnatune_com-magnatune_at_the_cc_salon-01-eyes_closed_the_seldon_plan-30-59.mp3'
}
```

## Pseudo Caption Example:

Input Tags: 

*"video game theme, no singer, instrumental, analog sounding, small keyboard, beatboxing, playful, cheerful, groovy"*

Output Pseudo Captions

*"instrumental track has a joyful and playful vibe, perfect for a video game theme. With no singer, the analog-sounding music features a small keyboard and beatboxing, creating a groovy and cheerful atmosphere"*

[More Information for pseudo caption generation](https://github.com/seungheondoh/lp-music-caps/blob/main/lpmc/llm_captioning/generate.py)
 

## Data Fields

| Name                         | Type            | Description                                                          |
|------------------------------|-----------------|----------------------------------------------------------------------|
| track_id                     | string          | Unique identifier for the track                                      |
| title                        | string          | Title of the song                                                    |
| artist_name                  | string          | Name of the artist performing the song                               |
| release                      | string          | Release name or album name of the song                               |
| tag_top50                    | list of strings | List of top 50 tags associated with the song                         |
| tag_top188                   | list of strings | List of top 188 tags associated with the song                        |
| caption_writing              | string          | Pseudo caption generated through a writing instruction               |
| caption_summary              | string          | Pseudo caption generated through a summary instruction               |
| caption_paraphrase           | string          | Pseudo caption generated through a paraphrase instruction            |
| caption_attribute_prediction | string          | Pseudo caption generated through an attribute_prediction instruction |
| pseudo_attribute             | list of strings | List of pseudo-attributes used in caption_attribute_prediction       |
| path                         | string          | File path or location of the audio clip                              |

## Data Splits

We used the full 188 tag vocabulary and did not generate captions for tracks that do not have associated tags (26k => 22k). 4K examples have empty tag and caption.

- train: 18706
- valid: 1825
- test: 5329
    
## Considerations for Using the Data

The LP-MusicCaps dataset is recommended to be used for research purposes. Due to the wrong labeling issue, we recommend not using caption_attribute_prediction and pseudo_attribute unless it is specifically for large-scale pretraining. Additionally, the field "is_crawled" indicates the samples used in the reference paper mentioned below.

## Discussion of Biases

It will be described in a paper to be released soon.

## Other Known Limitations

It will be described in a paper to be released soon.