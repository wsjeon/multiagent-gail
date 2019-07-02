# Multi-Agent Generative Adversarial Imitation Learning

Source code for our paper: [Multi-Agent Generative Adversarial Imitation Learning](https://arxiv.org/abs/1807.09936)

By [Jiaming Song](http://tsong.me), Hongyu Ren, Dorsa Sadigh, [Stefano Ermon](http://cs.stanford.edu/~ermon)

## Running the Code

- For code implementing MAGAIL, please visit `multiagent-gail` folder.
- For the OpenAI particle environment code, please visit `multiagent-particle-envs` folder.

## How to run
Run Multi-Agent ACKTR:
```
python -m sandbox.mack.run_simple
```

Run MAGAIL with Multi-Agent ACKTR:

```
python -m irl.mack.run_mack_gail [discrete]
```

Render results:

```
python -m irl.render
```

## Citation

If you find this code useful, please consider citing our paper:
```
@article{song2018multi,
  title={Multi-agent generative adversarial imitation learning},
  author={Song, Jiaming and Ren, Hongyu and Sadigh, Dorsa and Ermon, Stefano},
  year={2018}
}
```

