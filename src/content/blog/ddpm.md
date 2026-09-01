---
title: "Denoising Diffusion Probabilistic Models (DDPM)"
description: "What this blog is for, and what to expect."
date: 2026-08-25
tags: ["Generative AI", "text-to-image generation", "diffusion models", "ddpm"]
---


Denoising Diffusion Probabilistic Models (DDPM) [@ho2020ddpm] are modeled with a Markov chain, where each state represents the image at a certain timestep of the generation process. They learn to generate new data from random noise sampled from a normal distribution $\boldsymbol{\epsilon} \in \mathcal{N}(0, \mathbf{I})$ using two processes: the forward process and the reverse process. During training, the forward process is used to gradually add Gaussian noise to an image $\mathbf{x}_0$ over $T$ timesteps, obtaining a noisy image $\mathbf{x}_t$, where $0 < t < T$. Considering that $\mathbf{x}_0$ is a sample from the data distribution $q(\mathbf{x}0)$, we can characterize the forward process through the conditional probability of obtaining the next output given the current output, $q(\mathbf{x}t \mid \mathbf{x}{t-1})$. This probability follows a Gaussian distribution, with mean equal to $\sqrt{1-\beta_t}$ times the previous image $\mathbf{x}{t-1}$ and a standard deviation equal to $\beta_t$, since in each step we add Gaussian noise from $\mathcal{N}(0, I)$ to the previous image:

$$
q(\mathbf{x}t \mid \mathbf{x}{t-1})
= \mathcal{N}\left(
\mathbf{x}t;
\sqrt{1-\beta_t},\mathbf{x}{t-1},
\beta_t \mathbf{I}
\right)
\tag{1}
$$

where $\beta_t$ controls the amount of noise to add in each step. The image generated in step $t$ can be expressed through the diffusion process, where a noisy sample can be defined as:

$$
\mathbf{x}_t
= \sqrt{1-\beta_t},\mathbf{x}{t-1}

\sqrt{\beta_t},\boldsymbol{\epsilon}_t,
\tag{2}
$$

where $\boldsymbol{\epsilon}_t \sim \mathcal{N}(0, I)$ is the Gaussian noise added to the image in step $t$. Given the sample $\mathbf{x}_0$, if we recursively apply Equation 2 $t$ times, we obtain the image produced in step $t$ as a function of the initial sample $\mathbf{x}0$ and the noise $\boldsymbol{\epsilon}$ added in one step as follows:

$$
\mathbf{x}_t
= \sqrt{\bar{\alpha}_t},\mathbf{x}_0

\sqrt{1-\bar{\alpha}_t},\boldsymbol{\epsilon},
\tag{3}
$$

where $\alpha_t = (1-\beta_t)$ and $\bar{\alpha}t = \prod{s=1}^t \alpha_s = \prod{s=1}^t (1-\beta_s)$.

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

Recalling that diffusion is a Markov process, the next image $\mathbf{x}_t$ produced in step $t$ only depends on the current image $\mathbf{x}_{t-1}$. We can express the joint probability of the complete forward trajectory as:

$$
q(\mathbf{x}_{0})
= q(\mathbf{x}_0)
\prod{t=1}^T q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)
\tag{5}
$$

Assuming that we run the reverse diffusion process in a large number of steps $T$ and apply a small $\beta_t$ to the noise, the reverse process can also be characterized by a Markov chain with a Gaussian transition probability $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ in every step $t$. Therefore, the reverse process can be formulated as:

$$
p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}t\right)
= \mathcal{N}\left(
\mathbf{x}_{t-1};
\mu\theta\left(\mathbf{x}_t,t\right),
\Sigma\theta\left(\mathbf{x}_t,t\right)
\right)
\tag{6}
$$

where $\mu_\theta$ and $\Sigma_\theta$ are two functions parameterized by $\theta$, the neural network's parameters that must be learned. Again, because the reverse process is a Markov process, the next image $\mathbf{x}_{t-1}$ produced only depends on the current image $\mathbf{x}_t$. We express the joint probability of a complete reverse trajectory using Equation 7, where the initial image $p(\mathbf{x}_T)$ of the reverse trajectory follows an isotropic Gaussian distribution $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; 0, I)$ that does not depend on the model $\theta$.

$$
p_\theta\left(\mathbf{x}{0}\right)
= p\left(\mathbf{x}_T\right)
\prod{t=1}^T p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)
\tag{7}
$$

The authors proposed training the model to maximize the log-likelihood of the training data $\log p_\theta(\mathbf{x}0)$, or equivalently, minimize the negative log-likelihood $-\log p\theta(\mathbf{x}0)$. To simplify the training process, Ho et al. parameterized the training objective into the simplified expression in Equation 8, which yields an $\epsilon$-prediction neural network $\epsilon_\theta$ that predicts the noise $\epsilon$ added to the original image $\mathbf{x}_0$.

$$
L_{\mathrm{simple}}(\theta)
:= \mathbb{E}_{t,\mathbf{x}0,\epsilon}
\left[
\left|\epsilon - \epsilon\theta(\mathbf{x}_t,t)\right|^2
\right]
\tag{8}
$$

To sample from the trained model, we first draw a sample $\mathbf{x}_T$ of noise from the isotropic Gaussian distribution. Then, we iterate for $T$ steps, performing the reverse process, where we predict the noise that should be removed from $\mathbf{x}_t$ to push the sample closer to the data distribution. In each step, we random draw a noise sample $\mathbf{z}$ from the standard normal distribution and add it to the image. This helps preserve diversity instead of always following the conditional mean, and makes the reverse process match the learned reverse diffusion distribution. At the last timestep, $t = 0$, this noise is not added so the final image is not perturbed. This way, the reverse process can be imagined as if the initial $\mathbf{x}_T$ sample is following a trajectory predicted by our model $\epsilon_{\theta}$.

$$
\begin{aligned}
\mathbf{x}_{t-1}
&= \frac{1}{\sqrt{a_t}}
\left(
\mathbf{x}_t

\frac{1-a_t}{\sqrt{1-\bar{a}t}}
\epsilon\theta(\mathbf{x}_t,t)
\right)
+
\sigma_t\mathbf{z}
\end{aligned}
\tag{9}
$$
