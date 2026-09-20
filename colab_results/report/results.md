# Results — 20M LLM, 50M tokens, with and without reversibility

GPU **Tesla T4**, AMP **fp16**, 20,007,040 parameters (17,221,760 non-embedding), seq len 512.

## The three runs

| run | mode | batch (seqs) | tokens/step | steps | lr | final train loss | final val loss | tok/s (steady) | peak mem | model state | wall (train) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 · baseline | standard | 64 | 32,768 | 1526 | 0.001 | 1.9234 | 1.9492 | 62,435 | 6.05 GiB | 0.32 GiB | 13.4 min |
| 2 · reversible | leapfrog | 64 | 32,768 | 1526 | 0.001 | 1.9334 | 1.9606 | 47,387 | 1.12 GiB | 0.32 GiB | 17.6 min |
| 3 · reversible, max batch | leapfrog | 944 | 483,328 | 104 | 0.003 | 4.9305 | 4.9428 | 42,852 | 11.31 GiB | 0.32 GiB | 19.6 min |

At the same batch, reversibility cut peak memory 5.39x (activations+temporaries 5.74 → 0.81 GiB, 7.1x) and ran at 0.76x the baseline speed. Val loss moved by +0.0114.

![val loss](runs_val_loss.png)

![memory and speed](runs_mem_speed.png)

## Which reversible variant worked

Every integrator trained for 5M tokens at batch 64, same seed, same batches.

| mode | exact inverse? | status | final val loss | Δ vs standard | tok/s (steady) | speed vs standard | peak mem | grad cosine (init) | grad cosine (trained) |
|---|---|---|---|---|---|---|---|---|---|
| standard | (no inverse needed) | ok | 4.1126 | +0.0000 | 74,107 | 1.00x | 6.05 GiB | — | — |
| revnet | yes | ok | 4.1297 | +0.0171 | 55,181 | 0.74x | 1.16 GiB | 1.000000 | 0.999847 |
| midpoint | yes | ok | 4.0794 | -0.0332 | 51,018 | 0.69x | 1.12 GiB | 1.000000 | 0.999985 |
| leapfrog | yes | ok | 3.8611 | -0.2515 | 47,566 | 0.64x | 1.12 GiB | 0.999996 | 0.999947 |
| euler | no (fixed point) | ok | 5.7443 | +1.6317 | 22,171 | 0.30x | 1.08 GiB | 0.846343 | 0.009254 |


**Chosen variant: `leapfrog`**. Rule: lowest sweep val loss among variants whose trained-weight gradient cosine vs autograd is ≥ 0.999.

- `revnet`: val 4.1297, 55,181 tok/s
- `midpoint`: val 4.0794, 51,018 tok/s
- `leapfrog`: val 3.8611, 47,566 tok/s
- `euler`: excluded: grad cosine vs autograd (trained weights) 0.0093 < 0.999

![variant sweep](sweep_loss.png)

## Maximum batch size that fits

3 real training steps per attempt, seq len 512, GPU memory 14.6 GiB.

| mode | max batch (seqs) | tokens/step | peak at max | tok/s at max |
|---|---|---|---|---|
| standard | 160 | 81,920 | 13.58 GiB | 65,662 |
| ckpt | 896 | 458,752 | 14.22 GiB | 51,494 |
| leapfrog | 1200 | 614,400 | 14.28 GiB | 43,210 |

![probe](probe_memory.png)

## Samples (prompt: "Once upon a time", T=0.8, top-k 40)

**run1** — Once upon a time there was a little girl named Lily. One day, Lily went to the doctor with her mom many tools. Her mom asked her that she had to go to the doctor.

The doctor said that Lily was a dependable stranger and he would get out of her toys. Lily felt sad because she had no one to play with. Lily's mom explained to her that she needed to rest, so she gave her a hug.

While they were playing, Lily's mom asked her what was wrong. Lily replied, "I lost my toy. I did not have anything to play with."

**run2** — Once upon a time, there was a big lion who loved to eat carrots. One day, he went to the river to find some carrots for his carrots. But there was no one to eat all the carrots.

Suddenly, a little bird saw the lion in the sky. The lion asked the bird, "What's your name?" The bird replied, "My name is Bunny. I am a lion, but I am strong and you can play with me in the forest." The lion wanted to talk to the bird and said, "I am too small. You are a good monkey."

The

**run3** — Once upon a time. They loved was their ground in the go. They " day and a girl the girl she was a new the a't it was a little't friends. He, Timmy it to the time in to not help, had the girl he "Lily was a help a park.

 He would mom!" He went and the lot. She wanted. day.
Ben felt friends,, saw the girl. She's a play.
They the time namedDo his good her a girl was him to his little time and their little lot. They did a time they was the girl named
