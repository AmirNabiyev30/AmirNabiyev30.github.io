---
layout: post
title: "Gradient Descent Optimization with Momentum"
categories: "Machine_Learning"
permalink: /gradient-descent/
---

*Note: My knowledge comes heavily from the Deep Learning Book by Ian Goodfellow and Professors Gilbert Strang's lectures on youtube and finally I also assume that you understand basic calculus and some linear algebra*

For this blog post which also happens to be my first one, I wanted to talk about some of the fun math I've been doing with the CUNY Direct Reading Program or DRP for short

I've decided to focus on optimization for Machine Learning and I will eventually go talk about other ideas. I wanted to make this post so some people can see some interesting math and hopefully help me understand some concepts or correct my understanding.

To begin we need to try and gain a better understanding of gradient descent. Gradient descent uses the negative of the gradient vector to traverse a surrogate loss function which I'll denote as  $L(f(x;\theta), y)$. We use surrogate functions as usually the actual loss functions are harder to optimize and surrogate loss functions can capture the more general relationships in data and not just the ones in training data helping to prevent over-fitting.

Remember that $f(x;\theta)$ is our model guessing what y is based on our parameters $\theta$

The gradient vector can be expresed 
$$ 
\mathbf{g} \gets \nabla_{\theta}   \sum_{i=1}^{n} L(f(x_{i};\theta),y_i)
$$

You can imagine our loss function having a landscape and our negative gradient vector pointing to its valley or minimum and the area where the loss is the lowest is where the model is most accurate. In reality this landscape is high dimensional surface for which we can't visualize but thats where linear algebra shines

Now we can perform an analysis on how well we can converge to a minimum. First step is to understand the general taylor series expansion which looks as follows

$$
f(x) \approx f(x_0) + (x - x_0)^\top \mathbf{g} + \frac{1}{2}(x - x_0)^\top \mathbf{H} (x - x_0)
$$

We see the familiar the gradient vector which acts a first derivative but across all dimensions and our Hessian which is our second derivative but now across multiple dimensions. We can use taylor's remainder theorem to understand our error which will be the second term of the second taylor series but evaluated at an intermediary point.

Gradient descent iterates parameters by a step size of $-\epsilon\mathbf{g}$ as a way to prevent over-fitting. You can think of it as learning little bits at a time, this is why it's called the learning rate. Like for example when multiplying numbers you don't just focus on that specific example and remember the times tables for those specific numbers and what numbers carry over but the idea of carrying numbers over and fully understanding all the most common times tables and how to do multiplication generally

- Usually $\epsilon$ is always below 1 and greater than 0

So lets replace our x with iterative x that accounts for learning rate

$$
x = x_0 - \epsilon\mathbf{g}
\\f(x_0 - \epsilon\mathbf{g}) \approx f(x_0) -\epsilon\mathbf{g}^\top \mathbf{g} + \frac{1}{2}\mathbf{g}^\top \mathbf{H}\mathbf{g}
$$

*Note: Even though I'm using $f(x)$ which i previously mentioned is the model in this case I want to use f(x) for the loss function, just makes my life a little easier in typing*

Now we have a little more we can work with, specifically the hessian and understanding it.

Lets takes each term and understand each

- $f(x_0)$ this is is original loss computation

- $-\epsilon\mathbf{g}^\top \mathbf{g} $ is our basically our gradient vector dotted with itself times some $ \epsilon $ or learning rate, this will become some scalar

- $ \frac{1}{2}\mathbf{g}^\top \mathbf{H}\mathbf{g} $ is another term which can be considered as gradient but adjusted the curvature of the space. What do I mean by curvature, very similarly to how second derivatives in 2 dimensions represent concavity, think of expanding that to 3 dimensions, and overall you can think of it as the curvature of your space. This will also produce a scalar

Combining it all together, you can see why so many people continue to talk about the convex optimization, this is due to the fact that if function is convex, your hessian becomes largely negative helping you reach your minimum loss even faster. In truth, the loss functions of our models we try and optimize are not convex for which a good example of this is neural networks. 

When our loss function isn't convex is when the term with the Hessian can cause problems. When the Hessian causes problems it's usually called **Ill Conditioned** hessian. You can think of an bad hessian as one that represent's unsymmetrical part of curvature, a common example is a cliff, lots of curvature on one side and basically no curvature on the other side. It slows down training making it harder to optimize. So let's take a look at what we can do

### Momentum
*Note: This is my understanding coming from professor's Gilbert Strang's lecture on youtube for which you can access below. If I am wrong in any way feel free to email*
- [Accelerating Gradient Descent with Momentum](https://youtu.be/wrEcHhoJxjM?si=ZY78KZibBQNx51-O) 

The name comes from the physics approach and is pretty accurate to what actually happens and is actually quite simple. With regular gradient descent, each time step we compute a gradient based off our loss function at a certain set of parameters but I will keep it simple by just calling it position. Momentum proposes that we look back at our past gradient and take a portion of the direction that past gradient pointed at and combine it with our current gradient. Think of it as a look back term. Formally it can be written as

$$
\mathbf{g} \gets \nabla_{\theta}   \sum_{i=1}^{n} L(f(x_{i};\theta),y_i)\\
\mathbf{v} \gets \alpha\mathbf{v} - \epsilon\mathbf{g}\\
\theta \gets \theta + \mathbf{v}
$$

This stabilizes the learning a little bit allowing another besides the gradient and hessian to minimize our loss function. Let's take a look at it. So imagine again that we did a taylor series expansion and the Hessian term is our error term or correction term. Lets analyze how this term grows or shrinks and how we can control it.

### Analysis

$$
f(x) = \frac{1}{2}x^\top\mathbf{H}x 
$$

First thing we want to analyze is it's gradient, the gradient for this equation is a little weird but if we assume that $\mathbf{H}$ is symmetric the derivation results in

$$
\nabla f(x) = \mathbf{H}x
$$







