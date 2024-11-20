<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: Apache-2.0
-->

# MEL-PETs Defense for LLM Privacy Challenge

## Features

Code for the defense submission by the MEL-PETs team for the [NeurIPS 2024 LLM Privacy Challenge](https://llm-pc.github.io/) Blue Team track.

## Installation

To install `torch` and `unsloth`, we recommend following the installation guides provided by those libraries (as this should be customized based your specific CUDA setup, and may require manually installed further sub-dependencies, as instructed):

- https://pytorch.org/get-started/locally/
- https://docs.unsloth.ai/get-started/installation

### Environment setup

While we recommend following the installation guides for `torch` and `unsloth`, in order to properly install those packages with all of their necessary dependencies, as tailored for your specific environment.
However, here is a rough guide for an automated alternative:

```sh
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env
pip install -r requirements.txt
```
Then install LLM-PBE from https://github.com/QinbinLi/LLM-PBE

Note: the above `requirements.txt` does not list specific package versions and relies on pip to figure out the necessary CUDA environment automatically. If this is not working, you may need to select specific package and CUDA versions manually, as well as manually installing some sub-dependencies of these libraries (see the above installation guides for `torch` and `unsloth`).



## Usage

Download https://github.com/QinbinLi/LLMPC-Blue/blob/main/data/LLM-PC-development-scrubbed-data.jsonl to the current directory

1. fine-tuning (unlearning)

```sh
python unlearn.py --output_dir outputs_model
```

2. query the unlearned model with system prompt

```sh
python main.py
```


## Citation

If you use the software, please cite the following paper (note: currently under review for the competition):

```BibTeX
@inproceedings{melpets_blue,
    author = {Jing Liu and Ye Wang and Toshiaki Koike-Akino and Tsunato Nakai and Kento Oonishi and Takuya Higashi},
    title = {MEL-PETs Defense for the NeurIPS 2024 LLM Privacy Challenge Blue Team Track},
    booktitle = {NeurIPS 2024 LLM Privacy Challenge (under review)},
    year = 2024
}
```

## Related Links

- [NeurIPS 2024 LLM Privacy Challenge](https://llm-pc.github.io/)

## Contact

Jing Liu <jiliu@merl.com>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `Apache-2.0` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:

```
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
