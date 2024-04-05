---
title: Generative Modelling of Sequential Data
subtitle: M.Sc. thesis in collaboration with Corti ApS
insitute: Univ. Copenhagen Institute of Computer Science
author:
- Magnus Berg Sletfjerding
theme: Metropolis
# mainfont: "Noto Serif Light"
date: February 9th, 2022
# toc: true
aspectratio: 169
---

## Contents

```{=latex}
\tableofcontents
```

# Introduction

## Supervised and Unsupervised learning

```{=latex}
\begin{figure}
\centering
\resizebox{!}{0.5\textheight}{
\input{gfx/sup_unsup_tikz}
}\end{figure}
```

::: columns

:::: column

### Supervised Learning

Learns a mapping from data $\mathbf{x}$ to labels $\mathbf{y}$:  
$$
p(\mathbf{y}|\mathbf{x}) = \sum^N_{i=1} p(y_i| x_i)
$$

::::

:::: column

### Unsupervised Learning

Learns the structure of the data $\mathbf{x}$:
$$
p(\mathbf{x}) = \sum^N_{i=1} p(x_t)
$$

::::

:::

## Why study hierarchies of information in sequences?

- Most data we work with has some hierarchical structure

  - Text
  - Video
  - Proteins/DNA

- Human brains process hierarchies of information natively

  - Human-like AI requires hierarchical processing

- All real-world data has a sequential dimension - time! 

  

## Unsupervised sequence modeling 

Unsupervised sequence modeling optimize the likelihood $p(\cdot)$ of the data $\mathbf{x}$, calculated by conditioning the likelihood of $x_t$ on previous timesteps:
$$
p(\mathbf{x}) = \prod^N_{t=1}  p(x_t | x_{<t}) , \hspace{1cm}  \mathbf{x}\in \mathbb{R}^{N}
$$

## Recurrent vs. Convolutional Autoregressive models

```{=latex}
\begin{figure}[t]  
\centering 
  \begin{subfigure}[b]{0.45\linewidth}
  \resizebox{\columnwidth}{!}
    {
    \input{gfx/rnn_model_tikz}%
    }
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.45\linewidth}
  \resizebox{\columnwidth}{!}
    {
    \input{gfx/ar_model_tikz}%
    }
  \end{subfigure}
  %\caption{
%  Comparison between Recurrent and Autoregressive architectures for computing $\hat{x}_t$. Note that to estimate $\hat{x}_t$ accurately, the RNN needs to calculate its output multiple times, while the autoregressive model only needs to calculate its output once. 
  %}
\end{figure}  
```

::: columns

:::: column

### Recurrent architectures

Condition $p(x_t|x_{<t})$ through one or more hidden states $h_t$ passed between timesteps:
$$
p(x_t, h_t | x_{<t}) = p(x_t | x_{t-1}, h_{t-1})
$$



::::

:::: column

### Autoregressive Architectures

Condition  $p(x_t|x_{<t})$ by viewing a receptive field of size $R$ of the input sequence. 
$$
p(\mathbf{x}) = \prod^N_{t=R+1} p(x_t | x_{\geq t-R+1, <t})
$$

::::

:::



## WaveNet - Convolutional Autoregressive Sequence Modelling

```{=latex}

\begin{figure}
    \centering
    \resizebox{0.7\columnwidth}{!}{
    \input{gfx/wavenet_tikz}}
    %\caption{
    %Illustration of a 4-layer WaveNet architecture with exponentially increasing dilation $d_i=2^i, i\in [0, 3]$ and kernel size 2.
    %This results in a receptive field of size $2^4=16$. 
    %}
    %\label{fig:intro-wavenet}
\end{figure}

```



- Common vocoder in  Text to Speech production systems
- Makes use of dilated convolution to inflate receptive field 
- No "hidden state" for representing earlier timesteps
- Constrained to look back within receptive field

# Problem and Hypotheses

## Main Problem with WaveNet ... 

+ Local signal structure  
+ Missing long-range correlations
+ Low receptive field (300ms)
+ Generated audio sounds like babbling if not conditioned on phoneme or text representations 



## Hypotheses Investigated

1. WaveNet’s receptive field is the main limiting factor for modeling long-range dependencies. 
2. WaveNet’s stacked convolutional layers learn good representations of speech.
3. WaveNet’s hierarchical structure makes it suitable to learn priors over representations of speech such as text.
4.  A large WaveNet architecture trained on speech can generate coherent words and sentence fragments





# Experiments

## Experiment overview

1. Expanding Receptive Field by Stacking
2. Latent Space of Stacked WaveNets
3. WaveNet as an ASR preprocessor
4. WaveNet as a Language Model

## Expanding Receptive Field by Stacking - Setup

### Hypothesis tested

> 1. WaveNet’s receptive field is the main limiting factor for modeling long-range dependencies. 

### Setup

Transform x as:

```{=latex}
\begin{figure}
\centering
\resizebox{\columnwidth}{!}{
\input{gfx/stacking_tikz}
}\end{figure}
```





## Visualization of stacking on Sin curve

```{=latex}
\begin{figure}
\centering
\resizebox{1.0\columnwidth}{!}{

\input{gfx/sin_receptive_field_tikz.tex}
}
\end{figure}
```





## Expanding Receptive Field by Stacking - Results

```{=latex}
\begin{figure}
\centering
\resizebox{!}{0.9\textheight}{
\begin{tikzpicture}
\begin{axis}[
    title={WaveNet results on stacked audio samples},
    xlabel={Stacking ratio $S$},
    ylabel={TIMIT test set Likelihood [BPD]},
    %xmin=0, xmax=16,
    %ymin=10, ymax=13,
    ymajorgrids=true,
    grid style=dashed,
    legend pos=south east,
]
\addplot[
    color=blue,
    mark=square,
    ]
    coordinates {
    (2,11.835)(4, 12.331)(8, 12.621)(16, 13.060)
    };
    \addlegendentry{N=5 C=64}
\addplot[
    color=red,
    mark=square,
    ]
    coordinates {
    (2,11.878)(4, 12.249)(8, 12.503)(16, 12.861)
    };
    \addlegendentry{N=5 C=92}

\addplot[
    color=green,
    mark=square,
    ]
    coordinates {
    (2,12.156)(4, 12.312)(8, 12.740)(16, 13.042)
    };
    \addlegendentry{N=1 C=92}
    
\end{axis}
\end{tikzpicture}
}\end{figure}
```



## Expanding Receptive Field by Stacking - Conclusions

1. Stacking does not improve likelihoods significantly.
2. Increasing residual channels increases evaluation likelihoods.



Does this mean that the WaveNet does not extract any semantic information at all?



Is this a failure to measure the output correctly?

## Latent space of stacked WaveNet - Setup 

### Hypothesis tested

> 2. WaveNet’s stacked convolutional layers learn good representations of speech.

### Setup

1. Extract the hidden states of WaveNet for the TIMIT test set.
2. Reduce dimensionality from $C$ to 2 using Principal Component Analysis.
3. Plot density plot of all samples
4. Overlay 2500 samples with phoneme labels to observe clusters

## Latent space of stacked WaveNet - Results

```{=latex}
\begin{figure}
\includegraphics[width=0.5\columnwidth]{ 
        gfx/latent_exploration_PCA_S=2_C=64_N=5_L=10.png}%
\includegraphics[width=0.5\columnwidth]{
        gfx/latent_exploration_PCA_S=8_C=64_N=5_L=10.png
        }%
\end{figure}
```

## WaveNet as an ASR preprocessor - setup

### Hypothesis tested

> 2. WaveNet’s stacked convolutional layers learn good representations of speech.

### Idea

Are WaveNet's unsupervised representations more useful for Speech Recognition models than raw audio?

## WaveNet as an ASR preprocessor - setup (0 layers)



```{=latex}
\begin{figure}
\centering
\resizebox{\columnwidth}{!}{
\input{gfx/lstm_asr_tikz}
}
\end{figure}
```



## WaveNet as an ASR preprocessor - setup (3 layers)



```{=latex}
\begin{figure}
\centering
\resizebox{!}{0.9\textheight}{
\input{gfx/wavenet_asr_tikz}
}
\end{figure}
```



## WaveNet as an ASR preprocessor - Results



```{=latex}
\begin{figure}
\centering
\resizebox{!}{\textheight}{
\begin{tikzpicture}
\begin{axis}[
    title={WaveNet as ASR preprocessor},
    xlabel={Number of WaveNet Layers used},
    ylabel={CER on \texttt{clean-test}},
    ymajorgrids=true,
    grid style=dashed,
    legend style={at={(axis cs:0,0.95)},anchor=north west},
    legend cell align={left},
]
\addplot[
    color=blue,
    mark=square,
    ]
    coordinates {
    (0,1.00)(10, 1.0)(20, 0.9965)(30, 0.9983)(40, 0.9983)(50, 0.9983)
    };
    \addlegendentry{10min}
\addplot[
    color=red,
    mark=square,
    ]
    coordinates {
    (0,0.691)(10,0.7066)(20, 0.7969)(30, 0.8681)(40, 0.7326)(50, 0.6701)
    };
    \addlegendentry{1h}

\addplot[
    color=green,
    mark=square,
    ]
    coordinates {
    (0,0.6050)(10,0.5122)(20, 0.4618)(30, 0.4184)(40, 0.4392)(50, 0.4757)
    };
    \addlegendentry{10h}
    
\end{axis}
\end{tikzpicture}
}\end{figure}
```



## WaveNet as an ASR preprocessor - Conclusions

- Using WaveNet as a preprocessor decreases the loss when trained on the 1 hour and 10-hour training subsets. 

- The best performance occurs when using 30 layers of the WaveNet trained on 10 hours of training data.

- Notably, WaveNet's use as a preprocessor grows more competitive when increasing the training data size. 





## WaveNet as a Language Model - Setup 

### Hypothesis tested

> 3. WaveNet’s hierarchical structure makes it suitable to learn priors over representations of speech such as text.

### Setup

- WaveNet implemented as a character-level language model
- Categorical output distribution over alphabet
- Receptive field of 126 characters to match typical sentence length.





## WaveNet as a Language Model - Results 

```{=latex}
\begin{table}[htb]
    \centering
    \begin{tabular}{l|c||c}
        Model & Dataset & BPD (test) \\
        \hline
        Mogrifier LSTM \cite{melis_mogrifier_2020} & PTB & 1.083 \\
        Temporal Convolutional Network \cite{bai_empirical_2018} & PTB & 1.31 \\
        \hline
        WaveNet N=5 L=4 R=24 [RF 126] & PTB & 1.835 \\
        WaveNet N=5 L=4 R=32 [RF 126] & PTB & \textbf{1.666} \\
        WaveNet N=5 L=4 R=48 [RF 126] & PTB & 1.678 \\
        % WaveNet L=4 N=5 R=64 [RF 126] & Billion Word & 1.483 \\%1.677 \\
    \end{tabular}
\end{table}

```





## WaveNet as a Language Model - Conclusions 



- WaveNet remains below the bar compared to state-of-the-art models for Language Modelling, contradicting the hypothesis
- This may contribute to WaveNet's limited performance in speech synthesis

# Conclusions



| Hypothesis                                                   | Support? |
| ------------------------------------------------------------ | -------- |
| WaveNet’s receptive field is the main limiting factor for modeling long-range dependencies. | No       |
| WaveNet’s stacked convolutional layers learn good representations of speech. | Yes      |
| WaveNet’s hierarchical structure makes it suitable to learn priors over representations of speech such as text. | No       |
| A large WaveNet architecture trained on speech can generate coherent words and sentence fragments | No       |



## References

```{=latex}
\bibliography{vseq}
\bibliographystyle{abbrv}
```







# Appendix Slides

## Experiment: WaveNet Gradient analysis over input space - 1

### Explained

Run gradient evaluation over a trained WaveNet model and visualize the outputs. 

### Hypothesis tested 

**WaveNet uses the entirety of its receptive field for next-step prediction.**

- Gradients in the end of the RF (close to output) are larger than the gradients in the rest of the RF.

- Gradients do NOT collapse to 0 around the beginning of the RF (furthest away from output).

### Method

1. Calculate vector-Jacobian product with `torch.autograd`
2. Calculate norm with `torch.linalg.norm`

## Experiment: WaveNet Gradient analysis over input space - 2



```{=latex}
\begin{figure}[ht]
%        \resizebox{\columnwidth}{!}{
            \includegraphics[width=0.49\textwidth]{gfx/wavenet-8-4-64-epoch-70-gradients.png}%
            \includegraphics[width=0.49\textwidth]{gfx/wavenet-8-4-64-epoch-70-gradients-zoom.png}
 %       }
\end{figure}

```





## Notation

| Symbol                          | Explanation                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| $x_i$,$x_t$                     | The $i$th index of $\mathbf{x}$, of size $N$. $x_i \in \mathbb{R}^N$. $x_t$ is used when data is time-resolved. |
| $\mathbf{x}$                    | The data x, composed of vectors $x_i$. $\mathbf{x} \in \mathbb{R}^{T \times N}$ |
| $p_\theta(\cdot )$, $p(\cdot )$ | Likelihood function over model parameters $\theta$. Denoted $p(\cdot )$ for brevity |
| $\hat{x}_i$                     | Model prediction for $x_i$.                                  |
| $\mathcal{L}_{i}$               | Loss function for $i$th index.                               |
| $R$                             | Receptive field size.                                        |
| $S$                             | Size of stack size used in stack transformations             |
| $d_i$                           | Dilation of $i$th layer in a WaveNet architecture            |
| $C$                             | Number of residual channels                                  |

## Overview of Codebase and Experiments

::: columns

:::: {.column width=40%}

### Codebase

- https://github.com/JakobHavtorn/vseq/tree/wavenet-exps
- Collaborative codebase with Jakob Havtorn and Lasse Borgholt (PhDs at Corti)

- Includes custom implementations of likelihoods, data processing steps 

::::

:::: {.column width=60%}

### Work timeline

| Experiment                             | Time     |
| -------------------------------------- | -------- |
| Single Timestep WaveNet                | May      |
| WaveNet gradient evaluation check      | May-June |
| Audio LSTM control                     | May-June |
| Stacked WaveNet (softmax distribution) | June-Aug |
| Stacked WaveNet (DMoL)                 | Aug-Nov  |
| Latent space of stacked WaveNet        | Oct-Nov  |
| WaveNet as an ASR preprocessor         | Nov-Dec  |
| WaveNet as a Language Model (Text)     | Nov-Dec  |
| WaveNet as a Language Model (Nanobody) | Dec      |
|                                        |          |



::::

:::



## Overview of extra experiments

| Model                                    | Dataset(s)                                             | Notes                                          |
| ---------------------------------------- | ------------------------------------------------------ | ---------------------------------------------- |
| Single-Timestep WaveNet (softmax output) | TIMIT                                                  | Slow convergence compared to later DMoL        |
| Stacked WaveNet (softmax output)         | TIMIT, Librispeech                                     | Collapses to predict silence for all timesteps |
| Single Timestep WaveNet                  | Generated Sinusoids with periodically modulated pitch. | Fails to follow modulation in pitch            |

## Different tested embeddings for stacked WaveNet input

| Embedding type                                               | Dim   | Number       | Note                                                         |
| ------------------------------------------------------------ | ----- | ------------ | ------------------------------------------------------------ |
| Lookup table embedding with input dimensionality $S\times C$ | $128$ | $1024$       | Outputs collapsed to silence (suspect too sparse embeddings) |
| $S$ embeddings convolved together                            | 128   | $S\cdot 256$ | White noise output.                                          |
| 2 Layer perceptron with input size $S$ and output size $R$   | $R$   | Continuous   | Final used embedding                                         |



## Stacking Transformation

$$
x^*_t = \begin{pmatrix}
        x_{t} \\ \vdots \\ x_{t+S} \\
    \end{pmatrix},
    \hspace{1cm}
    t\in\{1,S+1,\dots,T-S\}, \mathbf{x}^*\in\mathbb{R}^{N/S \times S}\mathbf{x}\in\mathbb{R}^{N}
$$



## Residual Block of WaveNet 

```{=latex}
\begin{figure}
    \centering
    \resizebox{!}{0.9\textheight}{
    \input{gfx/wavenet_residual_block_tikz}
    %\caption{
    %Overview of WaveNet's residual block. 
    %The "Split/Copy" operation, representing the two configurations of the residual block is highlighted.
    %}
    %\label{fig:wavenet-res-block}
   }
\end{figure}
```



## Full WaveNet architecture

```{=latex}
\begin{figure}
    \centering
    \resizebox{!}{0.9\textheight}{
    \input{gfx/wavenet_extended_arch}
   }
\end{figure}
```





## DMoL vs. Categorical output distribution

### Discretized Mixture of Logistics

With a mixture of $K$ logistic distributions, for all discrete values of $x$ except edge cases:
$$
P(x|\pi, \mu, s) =CDF(x-0.5, x+0.5) = \sum_{i=1}^K \pi_i[\sigma(\frac{x +0.5 - \mu_i}{s_i}) - \sigma(\frac{x-0.5-\mu_i}{s_i}) ]
$$
Where $\sigma(\cdot )$ is the logistic sigmoid: $\sigma(x) = \frac{1}{1+e^x}$, $\pi$ is the relative weight vector, $\mu$ is the location vector and $s$ is the scale vector.  

### Softmax distribution

In a softmax distribution, the probability of the $i$th out of N discrete values is defined by:
$$
\sigma(\mathbf{x})_i = \frac{\exp(x_i)}{\sum_{j=1}^N \exp(x_j)}
$$

## Phonemes in TIMIT



```{=latex}

\begin{table}[htb]
    \centering
    \begin{tabular}{r|p{8cm}}
        Group & Phonemes \\
        \hline
        Vowels & \texttt{iy, ih, eh, ey, ae, aa, aw, ay, ah, ao, oy, ow, uh, uw, ux, er, ax, ix, axr, ax-h} \\
        Stops & \texttt{b, d, g, p, t, k, dx, q} \\
        Closures & \texttt{bcl, dcl, gcl, pcl, tck, kcl, tcl} \\
        Affricates & \texttt{jh, ch} \\
        Fricatives & \texttt{s, sh, z, zh, f, th, v, dh} \\
        Nasals & \texttt{m, n, ng, em, en, eng, nx} \\
        Semivowels and Glides & \texttt{l, r, w, y, hh, hv, el} \\
        Others & \texttt{pau, epi, h\#, 1, 2} \\ 
    \end{tabular}
    \caption{TIMIT phoneme groupings}
    \label{tab:timit-phonemes}
\end{table}

```



## Phoneme lengths in TIMIT

```{=latex}

\begin{figure}[t!]
    \centering
    \hfill
    \includegraphics[width=\columnwidth]{gfx/TIMIT_validation_set_phoneme_duration_boxplot_speaker_id_None.pdf}
    \caption{Boxplot of the duration of the pronunciation of phonemes in the TIMIT validation set.}
    \label{fig:timit-phoneme-duration}
\end{figure}
```





## Mu Law Distribution Illustrated

```{=latex}

\begin{figure}[ht]
    \centering
    \includegraphics[width=\columnwidth]{mu_law.png}
    \caption{
    Distribution of raw PCM values from the TIMIT test set. Far left: PCM (16-bit integers). Others: corresponding distribution after $\mu$-law encoding the waveform values with $\mu \in {8,10,16}$.
    %At 16 bits, all areas of the spectrum are covered while not overpopulated at the extreme bins.
    }
    %\label{fig:mu-law-enc}
\end{figure}
```



## Output distribution of WaveNet from Librispeech clean-100h - 1

```{=latex}
\begin{figure}
\resizebox{\columnwidth}{!}{
\centering
\includegraphics[width=0.45\textwidth]{gfx/S2_N5_L10_C64_output_distribution.png}
    \includegraphics[width=0.45\textwidth]{gfx/S4_N5_L10_C64_output_distribution.png}
}
\end{figure}

```

Sampled Output Distributions for WaveNet models trained on the Librispeech `clean-100h` subset.
    Distributions are in 16 bit $\mu$-law space and binned into 256 bins from -1 to 1.



## Output distribution of WaveNet from Librispeech clean-100h - 2

```{=latex}
\begin{figure}
\resizebox{\columnwidth}{!}{
\centering
    \includegraphics[width=0.45\textwidth]{gfx/S8_N5_L10_C64_output_distribution.png}
    \includegraphics[width=0.45\textwidth]{gfx/S16_N5_L10_C64_output_distribution.png}
}
\end{figure}

```

Sampled Output Distributions for WaveNet models trained on the Librispeech `clean-100h` subset.
    Distributions are in 16 bit $\mu$-law space and binned into 256 bins from -1 to 1.



## Output distribution of WaveNet from Librispeech clean-100h - constrained

```{=latex}
\begin{figure}
\resizebox{\columnwidth}{!}{
\centering
    \includegraphics[width=0.45\textwidth]{gfx/S2_N5_L10_C64_output_distribution_cut.png}

    \includegraphics[width=0.45\textwidth]{gfx/S16_N5_L10_C64_output_distribution_cut.png}
}
\end{figure}
```

## LSTM

 ```{=latex}
 \begin{figure}
 \resizebox{!}{0.9\textheight}{
 \centering
 \input{gfx/lstm_tikz}
 }
 \end{figure}
 ```



## Sigmoid and Tanh Activation functions

```{=latex}
\begin{figure}
\resizebox{\columnwidth}{!}{
\centering
\begin{tikzpicture}[declare function={dtanh(\x)=1/((exp(\x)+exp(-\x))^2;
}]
    \begin{axis}[
    title = {Tanh},
        axis lines = left,
        xlabel = \(x\),
        domain=-4:4,
        legend pos=north west,
    ]
    
    \addplot [mark=none,draw=blue,] {tanh(\x)};
    \addplot [mark=none,draw=red,dashed] {dtanh(\x)};

    \legend{$tanh(x)$,$sech^2(x)$}

\end{axis}
\end{tikzpicture}%
\hspace{0.15cm}
\begin{tikzpicture}[declare function={sigma(\x)=1/(1+exp(-\x));
sigmap(\x)=sigma(\x)*(1-sigma(\x));}]
    \begin{axis}[
    title = {Sigmoid $x \mapsto \frac{1}{1+\exp (-x)}$},
        axis lines = left,
        xlabel = \(x\),
        domain=-4:4,
        legend pos=north west,
    ]
    \addplot[blue,mark=none]   (x,{sigma(x)});
    \addplot[red,dashed,mark=none]   (x,{sigmap(x)});
    \legend{$\sigma(x)$,$\sigma'(x)$}

    
\end{axis}
\end{tikzpicture}%

}
\end{figure}
```



## ReLU activation function

```{=latex}
 \begin{figure}
\resizebox{!}{0.9\textheight}{
\centering
\begin{tikzpicture}
\begin{axis}[
    title = {ReLu},
        axis lines = left,
        xlabel = \(x\),
        domain=-4:4,
        legend pos=north west,
        ymin=-2,
    		ymax=5,
    ]
        \addplot+[mark=none,blue,domain=-4:0] {0};
        \addplot+[mark=none,blue,domain=0:4] {x};

        \addplot+[mark=none,red, dashed , domain=-4:0] {0};
        \addplot+[mark=none,red, dashed,domain=0:4] {1};


\end{axis}
\end{tikzpicture}%

}
\end{figure}
```

