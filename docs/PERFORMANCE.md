# Performance

What a day of paperwatch costs on one workstation, where the time goes, and how to measure your own setup. The
numbers were measured on 2026-11-21, most of them during the second one-day pilot (real arXiv and OpenAlex
papers, one topic).

## The machine and the model

| | |
| --- | --- |
| CPU | AMD Ryzen Threadripper PRO 3995WX, 64 cores / 128 threads |
| Memory | 251 GB |
| GPU | 2 × NVIDIA RTX A5000, 24 GB each |
| Model | Qwen3.8-27B, Q8_0 GGUF |
| Server | llama.cpp build 12373: `--ctx-size 262144 --parallel 1 --spec-type draft-mtp --spec-draft-n-max 5 --reasoning-effort medium --offline` |

## In short

- **The model is the bottleneck.** Reading one full-text paper took a median of **28 model-minutes** (7 to 209,
  over about 6 calls). If the model did nothing else, that is about 50 readings a day; the defaults allow 30
  screened papers plus up to 20 reference readings.
- **A third of the model's time went to rejected answers**: 2.95 of 9.4 hours were answers that failed the
  evidence checks or hit their token limit. This is the largest gain still available.
- **Everything else is cheap.** A collect run takes 40 s to 3 min, a cold screening of 1,600 new candidates about
  2 minutes, and the report a few seconds plus the writer's model call.

## Model calls by role

Successful calls during the pilot, 2026-11-21 17:24 to 2026-11-22 02:43 UTC:

| Role | Calls | Mean | Max | Prompt tokens (mean) | Answer tokens (mean, reasoning included) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `section_reader` | 46 | 5.3 min | 10.6 min | 12,138 | 6,202 |
| `reviewer` | 18 | 5.5 min | 11.4 min | 8,012 | 7,421 |
| `topic_assessor` | 18 | 2.3 min | 3.1 min | 3,919 | 2,085 |

All outcomes: 83 accepted answers (6.4 h), 26 rejected (2.6 h), and 2 cut off at their token limit (0.4 h).

Model minutes per completed review:

| Text read | Reviews | Median | Range |
| --- | ---: | ---: | ---: |
| Full text | 15 | 28 min | 7 to 209 min |
| Partial text | 2 | 31 min | 27 to 35 min |
| Abstract only | 1 | 3 min | |

The long tail is rejected answers, each of which costs a whole extra call. The slowest reading (209 min) included six
rejected section answers and one timeout, from the first pilot before its fixes; it was completed in the second.
(A review's minutes count every call of its job, including calls before the pilot started.)

## Speculative decoding

The same machine and model, reading a real 48,000-character section:

| llama-server setting | Generation | Prompt processing | One section |
| --- | ---: | ---: | ---: |
| no speculative decoding | 9.5 tokens/s | 290 tokens/s | about 13 min |
| `--spec-type draft-mtp --spec-draft-n-max 5` | 22.7 tokens/s | 195 tokens/s | about 5 min |

Answers with reasoning run to thousands of tokens, so generation speed dominates: drafting with the model's MTP heads
more than doubled throughput even though prompt processing got slower. Drafting 8 tokens instead of 5 was slower,
adding n-gram drafting gave no gain, and a smaller context size did not change the speed.

## Screening (CPU)

The pinned BAAI/bge-base-en-v1.5 model on ONNX Runtime embeds every new candidate on the CPU.

| Candidates | `screening.embedding_threads` | Time |
| ---: | ---: | --- |
| 455, cold | 8 | 60 s |
| 455, cold | 16 (default) | 30 s |
| 455, cold | 32 | 19 s |
| 455, embeddings cached | 16 | 0.3 s |
| 1,618, cold (a copy of the pilot's database) | 16 | 114 s wall, about 1,800 s of CPU, 2.7 GB peak memory |

Embeddings are cached by the hash of their input text, so only new or changed candidates cost time; a normal day
embeds just that day's arrivals. More threads make screening faster at the cost of cores the model server may want.
numpy runs with one BLAS thread (set in `paperwatch/__init__.py`). That made no difference here, since the
embedding model's own threads do the work, but it cut the test suite's CPU time 13 fold.

## Collection

The pilot's collect runs took 84 s, 39 s, and 174 s. The longest stage was following references (136 s of the
third run). Its time is set by the providers' rate limits (OpenAlex content requests at least 1.5 s apart, for
example), not by the CPU.

## Start-up and the test suite

| | Before (2026-11-21) | Now |
| --- | ---: | ---: |
| `paperwatch --help` | 0.365 s | 0.070 s |
| `paperwatch status` | 0.364 s | 0.301 s |
| `paperwatch monitor --json` | 0.412 s | 0.357 s |
| `uv run pytest` (wall, CPU) | 87 s, 642 s | 62 s, 26 s |

Each command now imports only its own code, so commands that don't need numpy, jsonschema, and httpx start faster.
The monitor's snapshot itself takes about 0.1 s per refresh, which happens every 15 seconds.

## Measuring your own setup

- `paperwatch --home <home> capacity --hours 24` shows, per role, call times and outcomes, minutes per review, and
  the estimated readings per day. From schema v11 on, it also shows llama.cpp's own timings: prompt and generation
  speed, the share of prompt tokens served from the server's cache, and how many drafted tokens were accepted.
- The monitor page's timeline shows every model call by role, with rejected answers hatched.
- To compare llama-server settings, let paperwatch run a day with each and compare `capacity`. Per-call numbers vary
  a lot with the paper, so judge by medians over many calls, not by a single paper.
