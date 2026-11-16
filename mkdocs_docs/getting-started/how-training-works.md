# How Training Works

*What "training a language model" means, from the first formula to the choices you make in a config. Each idea comes with the palingenesis command or key that controls it.*

---

## A model predicts the next token

A language model reads text as **tokens**: whole words, pieces of words, punctuation or special markers, each an integer id in a fixed **vocabulary** (about 150,000 entries for Qwen models). Given the tokens so far, one forward pass returns a probability for *every* token in the vocabulary being the next one. Generating text means picking one of them, appending it and running the model again.

What turns tokens into probabilities are the model's **weights** $\theta$: billions of numbers. Training changes nothing but the weights.

<figure class="pg-figure">
<div class="pg-figure__canvas">
<svg class="pg-fig" viewBox="0 0 700 196" role="img" aria-labelledby="fig-next-title fig-next-desc">
<title id="fig-next-title">From text to a probability over the next token</title>
<desc id="fig-next-desc">The text "The capital of Italy is" is split into five tokens, each with an integer id. The model, defined by its weights, reads the ids and outputs a probability for every token of the vocabulary being next: " Rome" 0.82, " a" 0.05, " the" 0.03, " located" 0.02, " home" 0.01, and the rest of the vocabulary shares what is left.</desc>
<defs>
<marker id="nx-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head" d="M0,0 L10,5 L0,10 z"/></marker>
</defs>
<text class="t-label" x="10" y="24">text</text>
<text class="t-label" x="166" y="24">tokens (ids)</text>
<text class="t-label" x="310" y="24">model</text>
<text class="t-label" x="470" y="24">p(next token)</text>
<rect class="box" x="10" y="80" width="130" height="58" rx="4"/>
<text x="22" y="104">The capital of</text>
<text x="22" y="124">Italy is</text>
<path class="line" d="M140,109 H162" marker-end="url(#nx-arrow)"/>
<rect class="box" x="166" y="42" width="118" height="22" rx="3"/><text class="t-mono" x="174" y="57">The</text><text class="t-mono t-muted" x="276" y="57" text-anchor="end">785</text>
<rect class="box" x="166" y="70" width="118" height="22" rx="3"/><text class="t-mono" x="174" y="85">␣capital</text><text class="t-mono t-muted" x="276" y="85" text-anchor="end">6722</text>
<rect class="box" x="166" y="98" width="118" height="22" rx="3"/><text class="t-mono" x="174" y="113">␣of</text><text class="t-mono t-muted" x="276" y="113" text-anchor="end">315</text>
<rect class="box" x="166" y="126" width="118" height="22" rx="3"/><text class="t-mono" x="174" y="141">␣Italy</text><text class="t-mono t-muted" x="276" y="141" text-anchor="end">15344</text>
<rect class="box" x="166" y="154" width="118" height="22" rx="3"/><text class="t-mono" x="174" y="169">␣is</text><text class="t-mono t-muted" x="276" y="169" text-anchor="end">374</text>
<path class="line" d="M284,109 H306" marker-end="url(#nx-arrow)"/>
<rect class="box-accent" x="310" y="42" width="128" height="134" rx="4"/>
<text x="374" y="102" text-anchor="middle">weights θ</text>
<text class="t-muted t-small" x="374" y="121" text-anchor="middle">billions of numbers</text>
<path class="line" d="M438,109 H460" marker-end="url(#nx-arrow)"/>
<text class="t-mono" x="470" y="57">␣Rome</text><rect class="fill-accent" x="546" y="47" width="114.8" height="12" rx="1"/><text class="t-mono t-muted" x="666" y="57">.82</text>
<text class="t-mono" x="470" y="81">␣a</text><rect class="fill-accent" x="546" y="71" width="7" height="12" rx="1"/><text class="t-mono t-muted" x="559" y="81">.05</text>
<text class="t-mono" x="470" y="105">␣the</text><rect class="fill-accent" x="546" y="95" width="4.2" height="12" rx="1"/><text class="t-mono t-muted" x="556" y="105">.03</text>
<text class="t-mono" x="470" y="129">␣located</text><rect class="fill-accent" x="546" y="119" width="2.8" height="12" rx="1"/><text class="t-mono t-muted" x="555" y="129">.02</text>
<text class="t-mono" x="470" y="153">␣home</text><rect class="fill-accent" x="546" y="143" width="1.4" height="12" rx="1"/><text class="t-mono t-muted" x="553" y="153">.01</text>
<text class="t-muted t-small" x="470" y="177">… the other ~150k tokens: .07</text>
</svg>
</div>
<figcaption>One forward pass. ␣ marks a leading space, which belongs to the token; ids and probabilities are illustrative. The model is defined entirely by its weights: the same weights give the same probabilities for the same tokens.</figcaption>
</figure>

## What training does

Training needs a number that says how wrong those probabilities are: the **loss**. For one position whose correct next token is $y$, the standard loss is the **cross-entropy**, the negative log of the probability the model gave to $y$:

$$
\ell = -\log p_\theta(y \mid \text{tokens before it})
$$

| $p_\theta(y)$ | 0.9 | 0.5 | 0.1 | 0.01 | uniform over 151,936 tokens |
|---|---|---|---|---|---|
| loss $-\log p$ | 0.11 | 0.69 | 2.30 | 4.61 | 11.93 |

A confident right answer costs almost nothing, a confident wrong one costs a lot, and a model that knows nothing (all tokens equally likely) sits at $\log V$ for a vocabulary of $V$ tokens. The training loss of a run is the mean over every trained token of the batch:

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{t \in T} \log p_\theta(x_t \mid x_{<t})
$$

where $x$ is the token sequence, $T$ the positions that are trained ([next section](#which-tokens-are-trained)) and $N$ their number.
{: .pg-where }

The **gradient** $\nabla_\theta \mathcal{L}$ gives, for every weight, how fast the loss changes when that weight moves: computed by **backpropagation** ("backward"), at about twice the cost of the forward pass. For cross-entropy it has a readable form at the output: the score of every token $v$ is pushed by $p_v - \mathbb{1}[v = y]$, so the correct token is pushed up by what it lacked ($1 - p_y$) and every other token down in proportion to the probability it took.

The **optimizer** then moves each weight a small step against its gradient. In its simplest form (gradient descent) with **learning rate** $\eta$:

$$
\theta \leftarrow \theta - \eta \, \nabla_\theta \mathcal{L}
$$

The default optimizer, AdamW (`train.optimizer: adamw`), uses a running average of each weight's gradient, divides it by the root of a running average of its square (so every weight moves at a similar pace), and adds weight decay (`train.weight_decay`), but the picture is the same: small steps downhill, as many as the data allows.

<figure class="pg-figure">
<div class="pg-figure__canvas">
<svg class="pg-fig" viewBox="0 0 700 200" role="img" aria-labelledby="fig-loop-title fig-loop-desc">
<title id="fig-loop-title">The training loop</title>
<desc id="fig-loop-desc">A batch of examples goes through the forward pass, which gives probabilities; the loss compares them with the correct tokens; the backward pass computes a gradient for every weight; the optimizer step moves the weights. With gradient accumulation, several micro-batches go through forward and backward, their gradients adding up, before one optimizer step. Then the next step starts.</desc>
<defs>
<marker id="lp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head" d="M0,0 L10,5 L0,10 z"/></marker>
</defs>
<path class="line" d="M489,72 V44 H69 V68" marker-end="url(#lp-arrow)" stroke-dasharray="4 3"/>
<text class="t-muted t-small" x="279" y="36" text-anchor="middle">next micro-batch: gradients add up (train.gradient_accumulation_steps)</text>
<rect class="box" x="10" y="72" width="118" height="56" rx="4"/>
<text x="69" y="96" text-anchor="middle">batch</text><text class="t-muted t-small" x="69" y="114" text-anchor="middle">examples from data</text>
<path class="line" d="M128,100 H148" marker-end="url(#lp-arrow)"/>
<rect class="box" x="150" y="72" width="118" height="56" rx="4"/>
<text x="209" y="96" text-anchor="middle">forward</text><text class="t-muted t-small" x="209" y="114" text-anchor="middle">probabilities</text>
<path class="line" d="M268,100 H288" marker-end="url(#lp-arrow)"/>
<rect class="box-accent" x="290" y="72" width="118" height="56" rx="4"/>
<text x="349" y="96" text-anchor="middle">loss</text><text class="t-mono" x="349" y="114" text-anchor="middle">−log p(y)</text>
<path class="line" d="M408,100 H428" marker-end="url(#lp-arrow)"/>
<rect class="box" x="430" y="72" width="118" height="56" rx="4"/>
<text x="489" y="96" text-anchor="middle">backward</text><text class="t-muted t-small" x="489" y="114" text-anchor="middle">a gradient per weight</text>
<path class="line" d="M548,100 H568" marker-end="url(#lp-arrow)"/>
<rect class="box" x="570" y="72" width="120" height="56" rx="4"/>
<text x="630" y="96" text-anchor="middle">optimizer step</text><text class="t-muted t-small" x="630" y="114" text-anchor="middle">weights move by η</text>
<path class="line" d="M630,128 V160 H69 V132" marker-end="url(#lp-arrow)"/>
<text class="t-muted t-small" x="349" y="178" text-anchor="middle">next step, until train.max_steps or train.epochs passes over the data</text>
</svg>
</div>
<figcaption>One training step. A step is one optimizer update; the <code>step=</code> counter in the log counts these. The micro-batches of a step are <code>train.per_device_batch_size</code> rows each, on every GPU.</figcaption>
</figure>

The learning rate is the most consequential number in a config. Too small and the loss barely moves; too large and each step jumps past the low point it was heading for:

<figure class="pg-figure">
<div class="pg-figure__canvas">
<svg class="pg-fig" viewBox="0 0 700 200" role="img" aria-labelledby="fig-gd-title fig-gd-desc">
<title id="fig-gd-title">Gradient descent on one weight, with a good and a too-large learning rate</title>
<desc id="fig-gd-desc">Two copies of the same bowl-shaped loss as a function of one weight. On the left, with a learning rate small enough, each step moves down the slope and the steps shrink as the slope flattens near the bottom. On the right, with a learning rate too large, each step jumps over the bottom to the other side, higher than before, and the loss grows.</desc>
<defs>
<marker id="gd-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="head-accent" d="M0,0 L10,5 L0,10 z"/></marker>
</defs>
<text class="t-label" x="35" y="20">learning rate small enough</text>
<text class="t-muted t-small" x="35" y="37">each step lowers the loss</text>
<text class="t-label" x="385" y="20">learning rate too large</text>
<text class="t-muted t-small" x="385" y="37">each step overshoots: the loss grows</text>
<text class="t-muted t-small" x="2" y="122">loss</text>
<text class="t-muted t-small" x="352" y="122">loss</text>
<polyline class="line" points="35,56 39,62.8 43,69.3 47,75.7 51,81.9 55,87.8 59,93.6 63,99.2 67,104.6 71,109.8 75,114.8 79,119.6 83,124.2 87,128.6 91,132.8 95,136.8 99,140.6 103,144.3 107,147.7 111,150.9 115,154 119,156.8 123,159.4 127,161.9 131,164.1 135,166.2 139,168.1 143,169.7 147,171.2 151,172.5 155,173.6 159,174.4 163,175.1 167,175.6 171,175.9 175,176 179,175.9 183,175.6 187,175.1 191,174.4 195,173.6 199,172.5 203,171.2 207,169.7 211,168.1 215,166.2 219,164.1 223,161.9 227,159.4 231,156.8 235,154 239,150.9 243,147.7 247,144.3 251,140.6 255,136.8 259,132.8 263,128.6 267,124.2 271,119.6 275,114.8 279,109.8 283,104.6 287,99.2 291,93.6 295,87.8 299,81.9 303,75.7 307,69.3 311,62.8 315,56"/>
<path class="line-accent" d="M53.2,78.2 Q84.1,60.2 115,145.9" marker-end="url(#gd-arrow-a)"/>
<path class="line-accent" d="M122.3,153.2 Q133.9,144.1 145.6,164.1" marker-end="url(#gd-arrow-a)"/>
<circle class="fill-accent" cx="50" cy="80.3" r="3.5"/>
<circle class="fill-accent" cx="118.8" cy="156.6" r="3.5"/>
<circle class="fill-accent" cx="149.7" cy="172.1" r="3.5"/>
<polyline class="line" points="385,56 389,62.8 393,69.3 397,75.7 401,81.9 405,87.8 409,93.6 413,99.2 417,104.6 421,109.8 425,114.8 429,119.6 433,124.2 437,128.6 441,132.8 445,136.8 449,140.6 453,144.3 457,147.7 461,150.9 465,154 469,156.8 473,159.4 477,161.9 481,164.1 485,166.2 489,168.1 493,169.7 497,171.2 501,172.5 505,173.6 509,174.4 513,175.1 517,175.6 521,175.9 525,176 529,175.9 533,175.6 537,175.1 541,174.4 545,173.6 549,172.5 553,171.2 557,169.7 561,168.1 565,166.2 569,164.1 573,161.9 577,159.4 581,156.8 585,154 589,150.9 593,147.7 597,144.3 601,140.6 605,136.8 609,132.8 613,128.6 617,124.2 621,119.6 625,114.8 629,109.8 633,104.6 637,99.2 641,93.6 645,87.8 649,81.9 653,75.7 657,69.3 661,62.8 665,56"/>
<path class="line-accent" d="M489.7,163 Q531.6,135.3 573.5,155.3" marker-end="url(#gd-arrow-a)"/>
<path class="line-accent" d="M574,155.1 Q516,120.2 457.9,141.2" marker-end="url(#gd-arrow-a)"/>
<path class="line-accent" d="M457.3,140.4 Q537.3,93.3 617.4,115" marker-end="url(#gd-arrow-a)"/>
<path class="line-accent" d="M618.1,113.3 Q508.2,44.5 398.3,66.6" marker-end="url(#gd-arrow-a)"/>
<circle class="fill-accent" cx="485" cy="166.2" r="3.5"/>
<circle class="fill-accent" cx="579" cy="158.1" r="3.5"/>
<circle class="fill-accent" cx="452.1" cy="143.5" r="3.5"/>
<circle class="fill-accent" cx="623.4" cy="116.7" r="3.5"/>
<circle class="fill-accent" cx="392.1" cy="67.9" r="3.5"/>
<text class="t-muted t-small" x="175" y="194" text-anchor="middle">one weight</text>
<text class="t-muted t-small" x="525" y="194" text-anchor="middle">one weight</text>
</svg>
</div>
<figcaption>Gradient descent on a single weight. Real models have billions of weights, so the bowl has billions of directions, but each step still follows the local slope. <code>train.learning_rate</code> sets η; the schedule (<code>train.lr_scheduler</code>, after <code>train.warmup_ratio</code> of warm-up from zero) changes it over the run.</figcaption>
</figure>

Run it over the data and you get a loss curve. Three shapes cover most of what you will see:

<figure class="pg-figure">
<div class="pg-figure__canvas">
<svg class="pg-fig" viewBox="0 0 700 226" role="img" aria-labelledby="fig-curves-title fig-curves-desc">
<title id="fig-curves-title">Three loss curves: healthy, learning rate too high, overfitting</title>
<desc id="fig-curves-desc">Left, healthy: the training loss falls quickly then flattens, noisy from step to step, with the evaluation loss slightly above it and following the same shape. Middle, learning rate too high: the loss first falls, then spikes repeatedly and climbs. Right, overfitting over three epochs: the training loss drops at each epoch boundary while the evaluation loss reaches a minimum during the second epoch and then rises; the checkpoint at that minimum is the one kept in best/.</desc>
<text class="t-label" x="34" y="20">healthy</text>
<text class="t-muted t-small" x="34" y="36">both fall, then flatten</text>
<text class="t-label" x="269" y="20">learning rate too high</text>
<text class="t-muted t-small" x="269" y="36">spikes, then the loss climbs</text>
<text class="t-label" x="504" y="20">overfitting</text>
<text class="t-muted t-small" x="504" y="36">eval turns up while train falls</text>
<text class="t-muted t-small" x="4" y="108">loss</text>
<text class="t-muted t-small" x="239" y="108">loss</text>
<text class="t-muted t-small" x="474" y="108">loss</text>
<path class="line-quiet" d="M564,48 V164"/><path class="line-quiet" d="M624,48 V164"/>
<text class="t-muted t-small" x="567" y="58">epoch 2</text><text class="t-muted t-small" x="627" y="58">epoch 3</text>
<path class="rule" fill="none" d="M34,40 V164 H214"/>
<polyline class="line-accent" points="34,58.2 36.5,61.4 39,68.3 41.5,71.4 44,75.9 46.5,85 49,89.5 51.5,86.4 54,94.8 56.5,98.4 59,95.1 61.5,102.4 64,102 66.5,107.6 69,108.6 71.5,114.8 74,112.8 76.5,112.8 79,117.5 81.5,117.3 84,119.4 86.5,126 89,121.5 91.5,124.1 94,127.8 96.5,131.1 99,125.1 101.5,129.4 104,128.2 106.5,127.7 109,129.8 111.5,128.8 114,133.9 116.5,131.1 119,134.7 121.5,131.1 124,132.1 126.5,139.2 129,139.3 131.5,139 134,133.1 136.5,137.9 139,136.7 141.5,139.7 144,138.3 146.5,139.6 149,140.1 151.5,138.4 154,138.6 156.5,136.1 159,138.2 161.5,136.3 164,137.1 166.5,136.1 169,138.9 171.5,143.3 174,137.6 176.5,136.8 179,137.5 181.5,140.4 184,139.3 186.5,143.6 189,138.4 191.5,140.7 194,143.2 196.5,145.1 199,138.5 201.5,137.5 204,145.1 206.5,139.2 209,142.5 211.5,144.7 214,143.5"/>
<polyline class="line" points="34,53.6 49,79.6 64,97.3 79,109.4 94,117.6 109,123.2 124,127 139,129.6 154,131.4 169,132.6 184,133.4 199,134 214,134.4" stroke-dasharray="5 3"/>
<path class="rule" fill="none" d="M269,40 V164 H449"/>
<polyline class="line-accent" points="269,54.1 271.5,59.6 274,71.3 276.5,72.4 279,81.4 281.5,81 284,87.9 286.5,87.7 289,90.5 291.5,97.1 294,96.5 296.5,104.2 299,108.4 301.5,106.9 304,113.1 306.5,113.8 309,114.1 311.5,114.3 314,119.1 316.5,121.3 319,116.6 321.5,121.8 324,118.2 326.5,119.7 329,124.3 331.5,124.6 334,124.9 336.5,124.7 339,125.2 341.5,125.2 344,97.9 346.5,121.1 349,124 351.5,124 354,120.6 356.5,124.2 359,122.7 361.5,114.9 364,92.1 366.5,117.1 369,121.7 371.5,116.1 374,112.9 376.5,118.2 379,109.6 381.5,81.5 384,113.7 386.5,104.8 389,105.3 391.5,100.8 394,103.6 396.5,70.4 399,94.5 401.5,103.2 404,100.9 406.5,89.6 409,56.6 411.5,92.3 414,87 416.5,82.6 419,85.1 421.5,82.2 424,80.9 426.5,77.7 429,71.3 431.5,74.5 434,70.7 436.5,60.8 439,72.7 441.5,59.7 444,53.9 446.5,59.9 449,59.1"/>
<path class="rule" fill="none" d="M504,40 V164 H684"/>
<polyline class="line-accent" points="504,58.6 506.5,67.8 509,72.9 511.5,75.6 514,78 516.5,86.2 519,88.3 521.5,91.8 524,91.4 526.5,97.4 529,97.3 531.5,104.9 534,106.3 536.5,106.4 539,107 541.5,112.3 544,115.9 546.5,118.5 549,117.3 551.5,121.4 554,118.5 556.5,119.8 559,125.8 561.5,126.4 564,132.2 566.5,134.6 569,134.5 571.5,138.8 574,141.3 576.5,141 579,138.2 581.5,142.7 584,142.9 586.5,143.1 589,143.7 591.5,144.4 594,141.3 596.5,147.4 599,149 601.5,142.7 604,143.6 606.5,143.2 609,147.6 611.5,144.2 614,144.8 616.5,150.2 619,146.7 621.5,146.3 624,156.2 626.5,157.5 629,159.4 631.5,159.3 634,160.3 636.5,161.6 639,158.1 641.5,160.4 644,156.8 646.5,161.6 649,156.5 651.5,158.1 654,156.6 656.5,156.9 659,157 661.5,159.4 664,161.6 666.5,158.4 669,161.6 671.5,161.6 674,159.2 676.5,160.3 679,161.2 681.5,157.4 684,159.9"/>
<polyline class="line" points="504,56 519,84.3 534,101.5 549,111.9 564,118.3 579,121.6 594,121.7 609,119.8 624,116.6 639,112.3 654,107.4 669,101.8 684,95.6" stroke-dasharray="5 3"/>
<circle class="box" cx="594" cy="121.7" r="4"/>
<text class="t-small" x="594" y="111" text-anchor="middle">best/</text>
<text class="t-muted t-small" x="124" y="180" text-anchor="middle">steps</text>
<text class="t-muted t-small" x="359" y="180" text-anchor="middle">steps</text>
<text class="t-muted t-small" x="594" y="180" text-anchor="middle">steps</text>
<path class="line-accent" d="M200,210 H226"/><text class="t-small" x="232" y="214">train loss (every step)</text>
<path class="line" d="M400,210 H426" stroke-dasharray="5 3"/><text class="t-small" x="432" y="214">eval loss (held-out data)</text>
</svg>
</div>
<figcaption>Schematic curves. The eval loss is measured on rows the model never trains on (<code>data.eval_dataset</code>, every <code>data.eval_every</code> steps); with it, palingenesis keeps the checkpoint with the lowest eval loss in <code>best/</code>. Without held-out data you cannot tell the right-hand case from the left-hand one.</figcaption>
</figure>

## Which tokens are trained

A chat dataset holds whole conversations, but you want the model to learn to *write the assistant's replies*, not to predict the user's questions or the system prompt. So the conversation is rendered with the model's own chat template, and only the assistant's tokens get a loss; everything else is **masked**: still read as context, never a target. This is the most important practical idea in post-training. If masking is wrong (the whole conversation trained, or nothing), the run looks normal and teaches the wrong thing.

<figure class="pg-figure">
<div class="pg-figure__canvas">
<svg class="pg-fig" viewBox="0 0 700 226" role="img" aria-labelledby="fig-mask-title fig-mask-desc" style="font-variant-ligatures: none">
<title id="fig-mask-title">Masking: in a rendered chat, only the assistant's reply is trained</title>
<desc id="fig-mask-desc">A conversation rendered with a ChatML chat template, as tokens. The system turn and the user turn, including their markers, are masked. In the assistant turn, the header tokens are masked and the reply tokens "It", "'s", " Rome", "." and the end-of-turn token are trained. The loss is the mean of minus log p over these five tokens, each predicted from all the tokens before it.</desc>
<text class="t-label" x="10" y="24">one conversation, rendered by the chat template</text>
<text class="t-muted t-small" x="10" y="61">system</text>
<rect class="box-quiet" x="112" y="46" width="92.8" height="22" rx="3"/><text class="t-mono t-muted" x="158.4" y="61" text-anchor="middle">&lt;|im_start|&gt;</text>
<rect class="box-quiet" x="208.8" y="46" width="51.4" height="22" rx="3"/><text class="t-mono t-muted" x="234.5" y="61" text-anchor="middle">system</text>
<rect class="box-quiet" x="264.2" y="46" width="23.8" height="22" rx="3"/><text class="t-mono t-muted" x="276.1" y="61" text-anchor="middle">\n</text>
<rect class="box-quiet" x="292" y="46" width="30.7" height="22" rx="3"/><text class="t-mono t-muted" x="307.4" y="61" text-anchor="middle">You</text>
<rect class="box-quiet" x="326.7" y="46" width="37.6" height="22" rx="3"/><text class="t-mono t-muted" x="345.5" y="61" text-anchor="middle">␣are</text>
<rect class="box-quiet" x="368.3" y="46" width="65.2" height="22" rx="3"/><text class="t-mono t-muted" x="400.9" y="61" text-anchor="middle">␣helpful</text>
<rect class="box-quiet" x="437.5" y="46" width="16.9" height="22" rx="3"/><text class="t-mono t-muted" x="446" y="61" text-anchor="middle">.</text>
<rect class="box-quiet" x="458.4" y="46" width="79" height="22" rx="3"/><text class="t-mono t-muted" x="497.9" y="61" text-anchor="middle">&lt;|im_end|&gt;</text>
<rect class="box-quiet" x="541.4" y="46" width="23.8" height="22" rx="3"/><text class="t-mono t-muted" x="553.3" y="61" text-anchor="middle">\n</text>
<text class="t-muted t-small" x="10" y="97">user</text>
<rect class="box-quiet" x="112" y="82" width="92.8" height="22" rx="3"/><text class="t-mono t-muted" x="158.4" y="97" text-anchor="middle">&lt;|im_start|&gt;</text>
<rect class="box-quiet" x="208.8" y="82" width="37.6" height="22" rx="3"/><text class="t-mono t-muted" x="227.6" y="97" text-anchor="middle">user</text>
<rect class="box-quiet" x="250.4" y="82" width="23.8" height="22" rx="3"/><text class="t-mono t-muted" x="262.3" y="97" text-anchor="middle">\n</text>
<rect class="box-quiet" x="278.2" y="82" width="37.6" height="22" rx="3"/><text class="t-mono t-muted" x="297" y="97" text-anchor="middle">What</text>
<rect class="box-quiet" x="319.8" y="82" width="23.8" height="22" rx="3"/><text class="t-mono t-muted" x="331.7" y="97" text-anchor="middle">'s</text>
<rect class="box-quiet" x="347.6" y="82" width="37.6" height="22" rx="3"/><text class="t-mono t-muted" x="366.4" y="97" text-anchor="middle">␣the</text>
<rect class="box-quiet" x="389.2" y="82" width="65.2" height="22" rx="3"/><text class="t-mono t-muted" x="421.8" y="97" text-anchor="middle">␣capital</text>
<rect class="box-quiet" x="458.4" y="82" width="30.7" height="22" rx="3"/><text class="t-mono t-muted" x="473.8" y="97" text-anchor="middle">␣of</text>
<rect class="box-quiet" x="493.1" y="82" width="51.4" height="22" rx="3"/><text class="t-mono t-muted" x="518.8" y="97" text-anchor="middle">␣Italy</text>
<rect class="box-quiet" x="548.5" y="82" width="16.9" height="22" rx="3"/><text class="t-mono t-muted" x="557" y="97" text-anchor="middle">?</text>
<rect class="box-quiet" x="569.4" y="82" width="79" height="22" rx="3"/><text class="t-mono t-muted" x="608.9" y="97" text-anchor="middle">&lt;|im_end|&gt;</text>
<rect class="box-quiet" x="652.4" y="82" width="23.8" height="22" rx="3"/><text class="t-mono t-muted" x="664.3" y="97" text-anchor="middle">\n</text>
<text class="t-muted t-small" x="10" y="133">assistant</text>
<rect class="box-quiet" x="112" y="118" width="92.8" height="22" rx="3"/><text class="t-mono t-muted" x="158.4" y="133" text-anchor="middle">&lt;|im_start|&gt;</text>
<rect class="box-quiet" x="208.8" y="118" width="72.1" height="22" rx="3"/><text class="t-mono t-muted" x="244.9" y="133" text-anchor="middle">assistant</text>
<rect class="box-quiet" x="284.9" y="118" width="23.8" height="22" rx="3"/><text class="t-mono t-muted" x="296.8" y="133" text-anchor="middle">\n</text>
<rect class="box-accent" x="312.7" y="118" width="23.8" height="22" rx="3"/><text class="t-mono" x="324.6" y="133" text-anchor="middle">It</text>
<rect class="box-accent" x="340.5" y="118" width="23.8" height="22" rx="3"/><text class="t-mono" x="352.4" y="133" text-anchor="middle">'s</text>
<rect class="box-accent" x="368.3" y="118" width="44.5" height="22" rx="3"/><text class="t-mono" x="390.6" y="133" text-anchor="middle">␣Rome</text>
<rect class="box-accent" x="416.8" y="118" width="16.9" height="22" rx="3"/><text class="t-mono" x="425.2" y="133" text-anchor="middle">.</text>
<rect class="box-accent" x="437.7" y="118" width="79" height="22" rx="3"/><text class="t-mono" x="477.2" y="133" text-anchor="middle">&lt;|im_end|&gt;</text>
<path class="rule" fill="none" d="M312.7,150 V156 H516.7 V150"/>
<text class="t-small" x="414.7" y="172" text-anchor="middle">loss = mean of −log p over these 5 tokens</text>
<rect class="box-accent" x="10" y="196" width="22" height="16" rx="3"/><text class="t-small" x="40" y="208">trained: a target of the loss</text>
<rect class="box-quiet" x="250" y="196" width="22" height="16" rx="3"/><text class="t-small" x="280" y="208">masked: read as context, never a target</text>
</svg>
</div>
<figcaption>A one-turn conversation in Qwen's ChatML template (token boundaries simplified; ␣ is a space). The end-of-turn token is trained too: that is how the model learns to stop.</figcaption>
</figure>

palingenesis derives the mask from the model's own chat template, so tool calls written by the assistant are trained, tool results are not (unless `data.include_observations: true`), and reasoning blocks are trained unless `data.train_on_reasoning: false`. Check it on your data before every new run, without a GPU:

```bash
pgs inspect --config sft.yaml --num_samples 3     # trained tokens in green, masked in gray
pgs validate --config sft.yaml --num_samples 200  # the same check over many rows
```

Details: [What gets loss](../guides/data.md#what-gets-loss-sft-masking) and, for multi-turn tool use, the [agentic guide](../guides/agentic-training.md).

## Pre-training and post-training

**Pre-training** runs the loop above on trillions of tokens of raw text, every token trained. The result is a **base model**: it continues text, but does not reliably answer, follow instructions or stop. **Post-training** is everything after: far less data, chosen to shape behaviour. It is what turns a base model into an assistant, a reasoning model or an agent, and what you do when you adapt such a model to your task. Most open models are published already post-trained (the "-Instruct" or chat versions); you usually continue from one of those. palingenesis is a post-training library (raw-text sources, `mode: pretrain`, cover continued pre-training on a domain).

<figure class="pg-figure">
<div class="pg-figure__canvas">
<svg class="pg-fig" viewBox="0 0 700 150" role="img" aria-labelledby="fig-stages-title fig-stages-desc">
<title id="fig-stages-title">Pre-training then post-training</title>
<desc id="fig-stages-desc">Random weights are pre-trained on trillions of tokens of raw text, every token trained, giving a base model that continues text. Post-training, the part palingenesis implements, uses examples, preferences, teachers and rewards, thousands to millions of rows, to give an assistant, reasoning or agent model.</desc>
<defs>
<marker id="st-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head" d="M0,0 L10,5 L0,10 z"/></marker>
</defs>
<rect class="box-quiet" x="10" y="52" width="94" height="64" rx="4"/>
<text x="57" y="80" text-anchor="middle">random</text><text x="57" y="97" text-anchor="middle">weights</text>
<path class="line" d="M104,84 H124" marker-end="url(#st-arrow)"/>
<rect class="box" x="126" y="52" width="180" height="64" rx="4"/>
<text x="138" y="74">pre-training</text>
<text class="t-muted t-small" x="138" y="92">raw text, trillions of tokens</text>
<text class="t-muted t-small" x="138" y="107">every token trained</text>
<path class="line" d="M306,84 H326" marker-end="url(#st-arrow)"/>
<rect class="box" x="328" y="52" width="104" height="64" rx="4"/>
<text x="380" y="80" text-anchor="middle">base model</text><text class="t-muted t-small" x="380" y="97" text-anchor="middle">continues text</text>
<path class="line" d="M432,84 H452" marker-end="url(#st-arrow)"/>
<text class="t-accent t-small" x="454" y="42">palingenesis</text>
<rect class="box-accent" x="454" y="52" width="236" height="64" rx="4"/>
<text x="466" y="74">post-training</text>
<text class="t-muted t-small" x="466" y="92">examples, pairs, teachers, rewards</text>
<text class="t-muted t-small" x="466" y="107">thousands to millions of rows</text>
<text class="t-muted t-small" x="466" y="138">→ an assistant, reasoning or agent model</text>
</svg>
</div>
<figcaption>Post-training can repeat: an instruct model is a post-trained model you can post-train again.</figcaption>
</figure>

## The post-training methods

The methods differ in two things: **which text the model trains on** (sequences fixed in advance, or the model's own samples generated during training) and **what signal each token gets**.

<figure class="pg-figure">
<div class="pg-figure__canvas">
<svg class="pg-fig" viewBox="0 0 700 262" role="img" aria-labelledby="fig-methods-title fig-methods-desc">
<title id="fig-methods-title">The post-training methods by text and by signal</title>
<desc id="fig-methods-desc">A grid. Columns: fixed data, written beforehand; and the model's own samples, generated while training. Rows: the signal. The next token of the data: SFT, pgs train, on fixed data. Chosen over rejected: DPO, pgs train with dpo.enabled, on fixed data. A teacher's distribution: off-policy distillation, pgs train with distill.enabled, on fixed data; on-policy distillation, pgs distill, on the model's own samples. A reward: RL, pgs rl, on the model's own samples.</desc>
<text class="t-label" x="10" y="36">signal per token</text>
<text class="t-label" x="210" y="20">fixed data</text>
<text class="t-muted t-small" x="210" y="36">written beforehand</text>
<text class="t-label" x="450" y="20">the model's own samples</text>
<text class="t-muted t-small" x="450" y="36">generated while training</text>
<text x="10" y="78">the data's next token</text>
<rect class="box" x="210" y="50" width="230" height="46" rx="4"/><text x="222" y="70">SFT</text><text class="t-mono t-muted" x="222" y="87">pgs train</text>
<rect class="box-quiet" x="450" y="50" width="240" height="46" rx="4"/><text class="t-muted" x="570" y="78" text-anchor="middle">—</text>
<text x="10" y="130">chosen over rejected</text>
<rect class="box" x="210" y="102" width="230" height="46" rx="4"/><text x="222" y="122">DPO</text><text class="t-mono t-muted" x="222" y="139">pgs train, dpo.enabled</text>
<rect class="box-quiet" x="450" y="102" width="240" height="46" rx="4"/><text class="t-muted" x="570" y="130" text-anchor="middle">—</text>
<text x="10" y="182">a teacher's distribution</text>
<rect class="box" x="210" y="154" width="230" height="46" rx="4"/><text x="222" y="174">off-policy distillation</text><text class="t-mono t-muted" x="222" y="191">pgs train, distill.enabled</text>
<rect class="box-accent" x="450" y="154" width="240" height="46" rx="4"/><text x="462" y="174">on-policy distillation</text><text class="t-mono t-muted" x="462" y="191">pgs distill</text>
<text x="10" y="234">a reward</text>
<rect class="box-quiet" x="210" y="206" width="230" height="46" rx="4"/><text class="t-muted" x="325" y="234" text-anchor="middle">—</text>
<rect class="box-accent" x="450" y="206" width="240" height="46" rx="4"/><text x="462" y="226">RL</text><text class="t-mono t-muted" x="462" y="243">pgs rl</text>
</svg>
</div>
<figcaption>Every method is the same loop with a different loss. The right-hand column samples from the model at every step, so it costs more per token, and is the only one that trains the model on the states its own mistakes lead to (<a href="#why-on-policy-methods-exist">below</a>).</figcaption>
</figure>

**SFT** (supervised fine-tuning) imitates examples: the cross-entropy above, on the assistant tokens of conversations you supply. It is how a model learns a format, a tool-calling style, a language register or a domain, and usually the first stage. [Data guide](../guides/data.md), [agentic guide](../guides/agentic-training.md).

**DPO** (direct preference optimization) learns from pairs: a prompt $x$, a chosen answer $y_w$ and a rejected one $y_l$. It raises the chosen answer's log-probability relative to the rejected one's, each measured against a frozen copy of the starting model $\pi_{\text{ref}}$:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\Big(\beta \big[\big(\log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x)\big) - \big(\log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x)\big)\big]\Big)
$$

where $\log \pi(y \mid x)$ is the sum of the answer's token log-probabilities, $\sigma$ the logistic function and $\beta$ `dpo.beta` (0.1).
{: .pg-where }

Use it when "better" is easier to show than to write: tone, length, when to refuse. [DPO guide](../guides/dpo.md).

**Off-policy distillation** trains on fixed sequences like SFT, but each trained token's target is a larger teacher's whole next-token distribution $p_T$ instead of one correct token. The default divergence is the forward KL, $\sum_v p_T(v) \log \frac{p_T(v)}{p_S(v)}$ for student distribution $p_S$; with a teacher that puts all its mass on the data's token it is exactly the cross-entropy, so SFT is the special case of a one-hot teacher. The extra signal is in the other tokens: how likely the teacher found each alternative. Teacher and student must share a vocabulary. [Off-policy distillation guide](../guides/offpolicy-distillation.md).

**On-policy distillation** reverses who writes: the student samples a completion, the teacher scores those exact tokens, and the loss is the reverse KL from student to teacher at every token of the student's own sample, $\sum_v p_S(v) \log \frac{p_S(v)}{p_T(v)}$. The student is corrected in the states it actually reaches, including the ones its mistakes create. Works across tokenizers and with several teachers. [On-policy distillation guide](../guides/distillation.md).

**RL** (reinforcement learning) needs no target text at all, only a way to score an answer: a checked math result, passing tests, a finished agent task. For each prompt the model samples a group of rollouts, each gets a reward $r_j$, and each rollout's tokens are reinforced in proportion to how much better than its group it did (and pushed down if it did worse):

$$
A_j = \frac{r_j - \bar r}{\sigma}, \qquad \text{update} \propto \sum_j A_j \sum_t \nabla_\theta \log \pi_\theta(y_{j,t} \mid y_{j,<t}, x)
$$

where $\bar r$ is the group's mean reward and $\sigma$ a standard deviation (by default the batch's: `loss.advantage_std`). This is GRPO's group-relative advantage; the default loss also clips the update.
{: .pg-where }

A group where every rollout scores the same gives no signal, so RL only helps on tasks the model already solves some of the time. [RL guide](../guides/rl.md).

**Merging** is not training: it combines the weights of checkpoints of the same model, in seconds and without a GPU. The common case pulls a fine-tune partway back toward its start, $W = W_{\text{start}} + \lambda (W_{\text{end}} - W_{\text{start}})$, to keep most of a gain and drop its side effects. [Merging guide](../guides/merging.md).

### Which one do I need?

| Goal | Method | Start | Run |
|---|---|---|---|
| Teach a format, a behaviour, a domain, from examples | [SFT](../guides/data.md) | `pgs init sft` | `pgs train --config sft.yaml` |
| Prefer one kind of answer over another (style, length, refusals) | [DPO](../guides/dpo.md) | `pgs init dpo` | `pgs train --config dpo.yaml` |
| Compress a bigger model with the same vocabulary, on a corpus | [Off-policy distillation](../guides/offpolicy-distillation.md) | `pgs init kd` | `pgs train --config kd.yaml` |
| Copy a teacher's reasoning or agent behaviour, fix the student's own errors and loops | [On-policy distillation](../guides/distillation.md) | `pgs init distill` | `pgs distill --config distill.yaml` |
| Get better at tasks you can check: math, code, tools, agents | [RL](../guides/rl.md) | `pgs init rl` | `pgs rl --config rl.yaml` |
| Keep part of a fine-tune, or add an expert, without training | [Merging](../guides/merging.md) | two checkpoints | `pgs merge interpolate START END --lam 0.3 -o OUT` |

A common order is SFT first, then DPO, on-policy distillation or RL on top, then a merge if the last stage cost something elsewhere. Each `pgs init` example is a complete config with small models and a public or example dataset: point the model and the data at yours.

## The knobs you will meet

Every value below is a config key ([configuration reference](../reference/configuration.md)); any of them can also be set on the command line, e.g. `pgs train --config sft.yaml --train.learning_rate 1e-5`.

| Knob | Keys (default) | What it means |
|---|---|---|
| Length of the run | `train.epochs` (1), `train.max_steps` (-1: from the epochs) | An epoch is one pass over the data. A streamed dataset has no known length, so it needs `max_steps`. Small datasets take more epochs ([how much data](first-training.md#how-much-data-do-you-need)). |
| Batch | `train.per_device_batch_size` (1), `train.gradient_accumulation_steps` (16) | Rows per optimizer step = batch size × accumulation × GPUs. Accumulation gives a large batch with the memory of a small one. |
| Learning rate | `train.learning_rate` (2e-5), `train.warmup_ratio` (0.05), `train.lr_scheduler` (cosine), `train.min_learning_rate` | The step size η and how it changes: up from 0 during warm-up, then down to the minimum. The shipped examples use 2e-5 for full SFT, 2e-4 for LoRA, 5e-7 for DPO and 2e-6 for RL: methods that start from a good model move it gently. |
| Gradient clipping | `train.max_grad_norm` (1.0) | A step whose gradient is longer than this is scaled down to it, so one bad batch cannot throw the weights far. |
| Sequence length | `data.max_seq_length` (8192) | The longest row, in tokens. Longer conversations are cut at turn boundaries. Memory for activations grows with it. `data.packing: true` fills each row with several short conversations. |
| Memory | `train.gradient_checkpointing` (selective), `memory.chunked_loss` (true), `memory.seco` (false) | Full fine-tuning with AdamW needs about 12 bytes per parameter before activations (bf16 weights and gradients, two fp32 optimizer states): 48 GB for a 4B model. Checkpointing recomputes activations in the backward pass instead of storing them; chunked loss never builds the full `[tokens × vocabulary]` logits; SeCO trains very long sequences chunk by chunk with exact gradients ([long sequences](../guides/long-sequences.md)). `pgs fit --config sft.yaml` measures the fastest setting that fits your GPU ([single GPU](../guides/single-gpu.md)). |
| Fewer trained weights | `plugins.custom: [{class: palingenesis.lora:LoRA, ...}]` | LoRA freezes the model and trains small low-rank adapters: far less optimizer memory, usually a little less learned ([LoRA guide](../guides/lora.md)). |
| Evaluation | `data.eval_dataset`, `data.eval_every` (100), `data.eval_samples` (200) | Held-out loss during training, and `best/`. |

**Eval loss is not quality.** A lower eval loss says the model predicts *your reference answers* better; it does not say the model got better at the task, and it says nothing about what it lost elsewhere. Pick the checkpoint with the loss, then judge it with real evaluations: the task's own checks (tests, exact answers), general benchmarks to catch regressions, and reading what the model writes. `data.eval_sources` tracks a separate held-out loss per capability, which catches some regressions early ([validation data](../guides/data.md#validation-data)).

## Reading your run

Every step logs one line (keep it: `pgs train --config sft.yaml 2>&1 | tee train.log`):

```
step=120 loss=1.3421 lr=1.98e-05 tok/s=6012 grad_norm=0.412 dt=1.21s eval=1.4012
```

| Field | Watch for |
|---|---|
| `loss` | falling fast, then slowly. Its absolute value depends on the data: compare runs on the same data only. |
| `grad_norm` | the gradient's length before clipping. Steady or slowly falling is normal; judge it against the run's own recent values, not a fixed number. |
| `lr` | the schedule: rising during warm-up, then decaying. |
| `tok/s`, `dt` | throughput and seconds per step. The first step is slow (compilation). |
| `eval` | on evaluation steps only: the held-out loss. |

`pgs monitor` reads that log and gives the state in one line (status, loss trend, throughput, time left); `pgs loss` analyses the whole curve:

```bash
pgs monitor --log_file train.log --brief --max_steps 500
# [HEALTHY] step=120 loss=1.3421 trend=-2.1% tok/s=6011 eta=7.7m
pgs loss --log_file train.log
```

| Sign | Likely cause | First thing to try |
|---|---|---|
| `loss=nan`, or loss climbing with `grad_norm` spikes | learning rate too high | lower `train.learning_rate` (÷3) |
| `SPIKE SKIPPED` lines | a few bad batches (the step is skipped, `train.spike_detection`) | many of them: look at the rows with `pgs inspect`, clean with `pgs prepare` |
| loss flat from the start | wrong masking, or learning rate too low | `pgs inspect`, then raise the rate |
| eval rising while train falls | overfitting | fewer epochs; use `best/` |
| train loss near zero | the data is memorised (many epochs, little data) | fewer epochs, more data |
| `tok/s` drops during the run | data loading cannot keep up | `data.num_workers`, `data.pretokenize` |

More signals (`health/*` metrics, warnings) are in [health monitoring](../reference/health.md); RL and on-policy distillation log their own (rewards, KL, rollout lengths), described in their guides and in [observability](../guides/observability.md).

## Going further

### Why fine-tuning makes models forget

The weights that hold a model's general skills are the same weights a fine-tune moves, so a gain on your task comes with drift elsewhere: instruction following, knowledge, other languages. This is **catastrophic forgetting**. It grows with the learning rate, the number of steps and how far the data is from what the model already does, and it tends to be most visible on small models. It does not show in the training loss: measure it with evaluations of what you want to keep. Three ways to limit it:

- **Replay:** mix general data into the run as another `data.sources` entry. Answers the model itself wrote to general prompts work better than a dataset's answers ([data guide](../guides/data.md#what-the-measurements-say)).
- **Interpolate back:** `pgs merge interpolate` toward the starting model with $\lambda$ around .3 to .5 keeps much of the gain and recovers much of what was lost ([merging](../guides/merging.md#choosing-lam-and-adopting-a-merge)).
- **Change less:** a lower learning rate, fewer steps, or LoRA.

### Why on-policy methods exist

SFT and off-policy distillation are *teacher-forced*: every prediction is made after a prefix from the data, never after the model's own words. At inference the model reads its own output, and one early mistake puts it in a state no training row contained, where the next mistake is more likely. This is **exposure bias**; it shows up as loops, drift and agents that do not recover from their own errors. On-policy distillation and RL train on the model's own samples, so those states are in the training data, and the teacher or the reward tells the model what to do there. Within SFT, `data.loop_recovery` and its companions show the model the way out of enumerated loop forms ([agentic guide](../guides/agentic-training.md), [on-policy distillation](../guides/distillation.md#the-students-own-episodes-training-where-it-starts-to-repeat-itself)).

### KL to a reference

A frozen copy of the starting model, the **reference** $\pi_{\text{ref}}$, measures how far training has moved the model. Per token, the usual measure is the KL divergence

$$
\mathrm{KL}(\pi_\theta \,\Vert\, \pi_{\text{ref}}) = \sum_v \pi_\theta(v) \log \frac{\pi_\theta(v)}{\pi_{\text{ref}}(v)}
$$

which is zero when the two distributions agree. Adding it to the loss trades some of the task signal for staying close to a model you know works. DPO has it built in ($\beta$ sets how strongly the reference holds; it is `dpo.reference_model`, by default the model as loaded). In `pgs rl` it is off by default, as in current published recipes: `loss.kl_ref: start` logs it as `policy/kl_ref`, and `loss.kl_coef` adds it to the loss. On-policy distillation's reverse KL has the same form with the teacher in place of the reference: the student is pulled toward a better model instead of held near its old self.

---

**Next:** [Your first real training](first-training.md) on your own data, then the guide of the method you picked above.
