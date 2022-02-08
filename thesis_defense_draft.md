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
p(\mathbf{x}|\mathbf{y}) = \sum^N_{i=1} p(y_i| x_i)
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

  

## Sequence modeling 

Sequence modeling optimize the model's likelihood $p(\cdot)$ over the data $\mathbf{x}$, by conditioning the probability of $x_t$ on previous timesteps:
$$
p(\mathbf{x}) = \prod^N_{t=1}  p(x_t | x_{<t}) , \hspace{1cm}  \mathbf{x}\in \mathbb{R}^N
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



- Common vocoder in Speech To Text production systems
- Makes use of dilated convolution to inflate receptive field 
- No "hidden state" for representing earlier timesteps
- Constrained to look back within receptive field

# Problem and Hypotheses

## Main Problem with WaveNet ... 

+ excelling at capturing local signal structure 
+ missing long-range correlations
+ low receptive field (300ms)
+ audio generation sounds like babbling 



## Hypotheses Investigated

1. WaveNet’s receptive field is the main limiting factor for modeling long-range dependencies. 
2. WaveNet’s stacked convolutional layers learn good representations of speech.
3. WaveNet’s hierarchical structure makes it suitable to learn priors over representations of speech such as text.
4.  A large WaveNet architecture trained on speech can generate coherent words and sentence fragments





# Experiments

## Experiment overview

1. Expanding Receptive Field by Stacking
2. Latent Space of Stacked WaveNets
3. WaveNet as a Language Model
4. WaveNet as an ASR preprocessor

## Expanding Receptive Field by Stacking - Setup

```{=latex}
\begin{figure}
\centering
\resizebox{\columnwidth}{!}{
\input{gfx/stacking_tikz}
}\end{figure}
```





## Expanding Receptive Field by Stacking - Results

```{=latex}
\begin{figure}
\centering
\resizebox{0.7\columnwidth}{!}{
\begin{tikzpicture}
\begin{axis}[
    title={WaveNet results on stacked audio samples},
    xlabel={Stacking ratio $S$},
    ylabel={Likelihood [BPD]},
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

1. Increasing the stacking does not improve likelihoods significantly.
2. Increasing the number of residual channels increases evaluation likelihoods.
3. 

## Latent space of stacked WaveNet - Setup 



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

## WaveNet as a Language Model - Setup 



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



## WaveNet as an ASR preprocessor - control setup



```{=latex}
\begin{figure}
\centering
\resizebox{\columnwidth}{!}{
\input{gfx/lstm_asr_tikz}
}
\end{figure}
```



## WaveNet as an ASR preprocessor - experiment setup



```{=latex}
\begin{figure}
\centering
\resizebox{0.9\columnwidth}{!}{
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





# Conclusions











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

## Overview of Codebase 

::: columns

:::: column

- Collaborative codebase with Jakob Havtorn and Lasse Borgholt (PhDs at Corti)
- Includes custom implementations of many modules in the 

::::

:::: column



::::

:::



TODO: Implementations to mention:

- Residual Stack
- Categorical WaveNet
- DMoL WaveNet



## Residual Block of WaveNet 

```{=latex}
\begin{figure}
    \centering
    \resizebox{!}{\textheight}{
    \input{gfx/wavenet_residual_block_tikz}
    %\caption{
    %Overview of WaveNet's residual block. 
    %The "Split/Copy" operation, representing the two configurations of the residual block is highlighted.
    %}
    %\label{fig:wavenet-res-block}
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


## Different tested embeddings for stacked WaveNet input

| Embedding type                                               | Dim   | Number       | Note                                                         |
| ------------------------------------------------------------ | ----- | ------------ | ------------------------------------------------------------ |
| Lookup table embedding with input dimensionality $S\times C$ | $128$ | $1024$       | Outputs collapsed to silence (suspect too sparse embeddings) |
| $S$ embeddings convolved together                            | 128   | $S\cdot 256$ | White noise output.                                          |
| 2 Layer perceptron with input size $S$ and output size $R$   | $R$   | Continuous   | Final used embedding                                         |



## Overview of extra experiments

| Model                                    | Dataset(s)                                             | Notes                                          |
| ---------------------------------------- | ------------------------------------------------------ | ---------------------------------------------- |
| Single-Timestep WaveNet (softmax output) | TIMIT                                                  | Slow convergence compared to later DMoL        |
| Stacked WaveNet (softmax output)         | TIMIT, Librispeech                                     | Collapses to predict silence for all timesteps |
| Single Timestep WaveNet                  | Generated Sinusoids with periodically modulated pitch. | Fails to follow modulation in pitch            |



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





## Mu Law Distribution 

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



## Visualization of stacking on Sin curve

```{=latex}
\begin{figure}
\centering
\resizebox{1.0\columnwidth}{!}{
\begin{tikzpicture}
\input{gfx/sin_receptive_field_tikz.tex}
}
\end{figure}
```

