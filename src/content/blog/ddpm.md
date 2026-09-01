---
title: "Denoising Diffusion Probabilistic Models (DDPM)"
description: "What this blog is for, and what to expect."
date: 2026-08-25
tags: ["Generative AI", "text-to-image generation", "diffusion models", "ddpm"]
---


Every generative model is trying to answer the same question, which is how to draw a new sample from a distribution we only ever observe through a finite set of examples. Denoising Diffusion Probabilistic Models (DDPM) [@ho2020ddpm] answer it with a slightly different approach. Instead of learning to produce an image in a single forward pass, the way a GAN or a VAE does, they learn to destroy one, and then run that destruction backwards.

The destruction is the easy half, and it involves no learning at all. We take a training image, add a small amount of Gaussian noise, and repeat that around a thousand times. Nothing of the original survives, and what we are left with is indistinguishable from noise drawn straight out of a normal distribution. Reversing it is the hard half. If a neural network can take one of those noisy images and undo a single step of the noising, we can start from a fresh sample of noise, apply the network over and over, and arrive at an image that was never in the training set.

Both directions are modeled as Markov chains, where each state is the image at a given timestep and every transition depends only on the state immediately before it. That assumption is what keeps the problem tractable, because the network never has to reason about an entire trajectory, only about a single step of one. From here we look at the forward process that adds the noise, the reverse process that removes it, the objective used to train the network, and finally the sampling loop that turns noise into an image.

## The forward process

During training, the forward process is used to gradually add Gaussian noise to an image $\mathbf{x}_0$ over $T$ timesteps, obtaining a noisy image $\mathbf{x}_t$, where $1 \le t \le T$. Considering that $\mathbf{x}_0$ is a sample from the data distribution $q(\mathbf{x}0)$, we can characterize the forward process through the conditional probability of obtaining the next output given the current output, $q(\mathbf{x}t \mid \mathbf{x}{t-1})$. This probability follows a Gaussian distribution, with mean equal to $\sqrt{1-\beta_t}$ times the previous image $\mathbf{x}{t-1}$ and a variance equal to $\beta_t$, since in each step we add Gaussian noise from $\mathcal{N}(0, \mathbf{I})$ to the previous image:

$$
q(\mathbf{x}t \mid \mathbf{x}{t-1})
= \mathcal{N}\left(
\mathbf{x}t;
\sqrt{1-\beta_t},\mathbf{x}{t-1},
\beta_t \mathbf{I}
\right)
\tag{1}
$$

where $\beta_t$ controls the amount of noise to add in each step. Notice that the mean is not simply the previous image. The factor $\sqrt{1-\beta_t}$ shrinks the signal that is already there at the same time as the noise is added, and that is why the chain converges to pure noise instead of to an image buried under an ever growing pile of it. The image generated in step $t$ can be expressed through the diffusion process, where a noisy sample can be defined as:

$$
\mathbf{x}_t
= \sqrt{1-\beta_t},\mathbf{x}{t-1}

\sqrt{\beta_t},\boldsymbol{\epsilon}_t,
\tag{2}
$$

where $\boldsymbol{\epsilon}_t \sim \mathcal{N}(0, \mathbf{I})$ is the Gaussian noise added to the image in step $t$. Given the sample $\mathbf{x}_0$, if we recursively apply Equation 2 $t$ times, we obtain the image produced in step $t$ as a function of the initial sample $\mathbf{x}_0$ and a single noise sample $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$. This collapse into one draw is possible because a sum of Gaussians is itself Gaussian, so the noise accumulated over the $t$ steps can be summarized by one sample:

$$
\mathbf{x}_t
= \sqrt{\bar{\alpha}_t},\mathbf{x}_0

\sqrt{1-\bar{\alpha}_t},\boldsymbol{\epsilon},
\tag{3}
$$

where $\alpha_t = (1-\beta_t)$ and $\bar{\alpha}t = \prod{s=1}^t \alpha_s = \prod{s=1}^t (1-\beta_s)$.

This is the equation that makes diffusion models practical to train. To obtain a sample at timestep $t$ we never have to run the chain $t$ times, since one draw of $\boldsymbol{\epsilon}$ and one weighted sum already land us at $\mathbf{x}_t$. It also gives a clean reading of the noise schedule through $\bar{\alpha}_t$, which starts close to $1$, where $\mathbf{x}_t$ is almost all signal, and decays towards $0$, where $\mathbf{x}_T$ becomes indistinguishable from the noise we will later sample at generation time.

According to the VAE reparameterization trick, a random variable $\mathbf{z}$ following a Gaussian distribution $\mathbf{z} \sim \mathcal{N}(\mathbf{z}; \mu, \sigma^2 \mathbf{I})$ can be replaced by $\mu + \sigma\boldsymbol{\epsilon}$, where $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{\epsilon}; 0, \mathbf{I})$. Considering this reparameterization trick, Equation 3 proves that the distribution $q(\mathbf{x}_t \mid \mathbf{x}_0)$ is the Gaussian and is written as:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0)
= \mathcal{N}\left(
\mathbf{x}_t;
\sqrt{\bar{\alpha}_t},\mathbf{x}_0,
\left(1-\bar{\alpha}_t\right)\mathbf{I}
\right)
\tag{4}
$$

Recalling that diffusion is a Markov process, the next image $\mathbf{x}_t$ produced in step $t$ only depends on the current image $\mathbf{x}_{t-1}$. We can express the joint probability of the complete forward trajectory $\mathbf{x}_{0:T} = \left(\mathbf{x}_0, \ldots, \mathbf{x}_T\right)$ as:

$$
q(\mathbf{x}_{0:T})
= q(\mathbf{x}_0)
\prod{t=1}^T q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)
\tag{5}
$$

Nothing in this equation is learned. The forward process is fixed in advance by the schedule $\beta_1, \ldots, \beta_T$, so we can use it freely as a noise-generating device during training. For any image and any timestep we already know exactly what the answer should be.

## The reverse process

To generate an image, we need to walk this chain backwards, which means sampling from the reverse conditional $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$. This distribution is intractable, since computing it would require knowing the entire data distribution. However, if the forward process runs for a large number of steps $T$ and adds only a small amount of noise in each one (a small $\beta_t$), the true reverse conditional is itself approximately Gaussian. This is what allows us to approximate it with a Gaussian whose mean and covariance are predicted by a neural network:

$$
p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}t\right)
= \mathcal{N}\left(
\mathbf{x}_{t-1};
\mu\theta\left(\mathbf{x}_t,t\right),
\Sigma\theta\left(\mathbf{x}_t,t\right)
\right)
\tag{6}
$$

where $\mu_\theta$ and $\Sigma_\theta$ are two functions parameterized by $\theta$, the neural network's parameters. In practice, Ho et al. do not learn the covariance. They fix it to $\Sigma_\theta(\mathbf{x}_t, t) = \sigma_t^2 \mathbf{I}$, where $\sigma_t^2$ is set to either $\beta_t$ or $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$, leaving only the mean to be learned. All of the model's capacity therefore goes into one task, repeated once per step, which is to look at a noisy image and its timestep and describe the slightly cleaner image it most likely came from. Again, because the reverse process is a Markov process, the next image $\mathbf{x}_{t-1}$ produced only depends on the current image $\mathbf{x}_t$. We express the joint probability of a complete reverse trajectory using Equation 7, where the initial image $p(\mathbf{x}_T)$ of the reverse trajectory follows an isotropic Gaussian distribution $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; 0, \mathbf{I})$ that does not depend on the model $\theta$.

$$
p_\theta\left(\mathbf{x}_{0:T}\right)
= p\left(\mathbf{x}_T\right)
\prod{t=1}^T p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)
\tag{7}
$$

## Training

The authors proposed training the model to maximize the log-likelihood of the training data $\log p_\theta(\mathbf{x}0)$, or equivalently, minimize the negative log-likelihood $-\log p\theta(\mathbf{x}0)$. This likelihood is intractable, so, as in VAEs, the model is trained on a variational lower bound of it instead. Because both the forward posterior and the reverse transitions are Gaussian, that bound decomposes into a sum of KL divergences between Gaussians, one per timestep, which have a closed form. Rewriting the mean $\mu_\theta$ in terms of the noise through Equation 3, and dropping the weighting coefficients that each term carries, Ho et al. arrive at the simplified objective in Equation 8, which yields an $\boldsymbol{\epsilon}$-prediction neural network $\boldsymbol{\epsilon}_\theta$ that predicts the noise $\boldsymbol{\epsilon}$ added to the original image $\mathbf{x}_0$.

$$
L_{\mathrm{simple}}(\theta)
:= \mathbb{E}_{t,\mathbf{x}0,\boldsymbol{\epsilon}}
\left[
\left|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\right|^2
\right]
\tag{8}
$$

where $t$ is drawn uniformly from $\{1, \ldots, T\}$, $\mathbf{x}_0$ is an image from the training data, and $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ is the noise used to build $\mathbf{x}_t$ through Equation 3. Training therefore reduces to picking a random image and a random timestep, noising that image in a single shot, and asking the network to name the noise. There is no adversarial game here and no likelihood to evaluate at training time, only a regression loss against a target we generated ourselves, which is a large part of why diffusion models train so stably.

## Sampling

To sample from the trained model, we first draw a sample $\mathbf{x}_T$ of noise from the isotropic Gaussian distribution. Then, we iterate for $T$ steps, performing the reverse process, where we predict the noise that should be removed from $\mathbf{x}_t$ to push the sample closer to the data distribution. In each step, we randomly draw a noise sample $\mathbf{z}$ from the standard normal distribution and add it back to the image, scaled by $\sigma_t$. This helps preserve diversity instead of always following the conditional mean, and makes the reverse process match the learned reverse diffusion distribution. At the final iteration, $t = 1$, which produces $\mathbf{x}_0$, this noise is not added ($\mathbf{z} = 0$) so the final image is not perturbed. This way, the reverse process can be imagined as if the initial $\mathbf{x}_T$ sample is following a trajectory predicted by our model $\boldsymbol{\epsilon}_\theta$.

$$
\begin{aligned}
\mathbf{x}_{t-1}
&= \frac{1}{\sqrt{a_t}}
\left(
\mathbf{x}_t

\frac{1-a_t}{\sqrt{1-\bar{a}t}}
\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)
\right)
+
\sigma_t\mathbf{z}
\end{aligned}
\tag{9}
$$

Equation 9 is easiest to read from the inside out. $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)$ is the network's guess at all the noise present in $\mathbf{x}_t$. Scaling it by $\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}$ and subtracting it strips away only the share of that noise which belongs to the current step, not the whole of it. Dividing by $\sqrt{\alpha_t}$ then undoes the shrinking that the forward process applied to the signal. What is left is the mean of $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$, and adding $\sigma_t\mathbf{z}$ turns that mean into an actual sample. Run this $T$ times and the noise we started from has become an image.
