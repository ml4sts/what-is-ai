(what-is-ai)=
# What *is* AI? 

Goals:
- define AI in multiple ways and contextualize the term usage
- describe *how* modern AI is built and how that differs from older algorithms
- 

## AI *is* computer science

Building {term}`artificial intelligence` has been the goal of computing research (computing is a  broader feild including both computer *sicence* or computer *engineering*) the whole time.  Computing is not a very old discipline, but can trace its roots to math and electricial engineering (and through that, physics). Computer Science has espeically been focused on this as a goal. 


AI is a field of study, which means there is a community of people egaged in this work. 

{term}`AI <artificial intelligence>` can be done in many ways and has been over time, but the current dominant paradigm is machine learning.  
The formal definitions of AI are broad and in that way, almost everything in a computer can be considered to "be AI" but, in this moment, AI typically refers to a few things:
- {term}`Large Language Models <large language model>` (and multimodal mdoels with the same basic goal) that produce text and accomplish high level goals based on english
- {term}`predictive models <predictive model>` that predict specific things or automatically label things and are the result of machine learning
- complex systems that combine the two things above with additional logic


What is common across all of these, and all things in computer science, is {term}`algorithms <algorithm>`. Algorithms have gotten a lot more attention recently, but they are not fundamentally new, mathematicians have developed and studied them formally for centuries and informally, people have developed them all over. 

## How Algorithms are Made 

A familiar way to think about what an {term}`algorithm` is, as a recipe. A recipe consists of a set of ingredients (inputs) and a set of instructions to follow (procedures) to produce a specific dish (output)

Mathematicians have developed algorithms for centuries by:
1. Picking a problem to solve in the world
2. Writing an approximation for the relevant part of the world that they can write down in mathematical language
3. Working to solve the problem in the mathematical space
4. Documenting their process so that it is repeatable

Computer Scientists do the same, with the main change being that the solution is expressed in a {term}`programming language` instead of in a spoken language, for example `Python` instead of English. 

The challenge with this is as we approach more complex problems as our goal to try to delegate to a computer, the approximaitons we make get in the way more. Writing an algorithm to add numbers together is straightforward, writing an algorithm to detect if a set of pixels represents a person(eg if there is a person in front of a self driving car) is much more complex. 

This is where {term}`machine learning` comes in. 

:::{important}
Current success in AI is based on machine learning
:::


## What is ML? 

In machine learning, we change the process a little.  

Instead of solving the problem and figuring out precise steps to carry out, the people define a generic strategy, collect a lot of examples, and write a {term}`learning algorithm` to fill in the details of the generic strategy from the examples. 
Learning algorithms are developed the way we have always developed algorithms, but then these algorithms essentially write a prediction or inference algorithm that is what gets sent to the world. 


All ML consists of two parts: learning and prediction 

these can also go by other names. 

::::{grid} 1 1 2 2

:::{card}
:header: Learning may be called:
- fitting
- optimization
- training
:::

:::{card}
:header: Prediction may be called:
- inference
- testing
:::

::::


### A common assumption


All ML has some sort of underlying assumption, almost all ML relies on two key assuptions, that can be written in many ways: 

::::{tab-set}
:::{tab-item} Plain English
:sync: english 
1. A relationship exists to that we can determine, or predict `outcome or target`  using `input`. 
2. Given enough examples a computer can find that relationship

where: 
- `outcome or target` is the goal of the task 
- `input` 
:::
:::{tab-item} Math
:sync: math

1. $\exists f,\theta$ such that $y = f(x; \theta)$
2. $\exists \mathcal{A}$ such that $f_{\theta} = \mathcal{A}((x,y)^D)$

where:
- $x$ is an input or {term}`features <feature>`
- $y$ is an output, {term}`label`, or {term}`target`
- $f$ is a relationship between them, a function of some general class
- $\theta$ is a set of parameters to make the function specific
- $\mathcal{A}$ is an algorithm
- our dataset $\mathcal{D}$ is comprised of $D$ $(x,y)$ pairs


:::
:::{tab-item} Code (Python)
:sync: python


```
output = my_fun(input, parameters)
```

:::
:::{tab-item} Diagram
:sync: 

```{mermaid}
flowchart LR
    x(input <br>x) -->  f[f ] --> y(output <br>y)
```
<!--  <sub>#952;</sub -->


:::

::::


<!-- To make this concrete, this could be as simple as a linear regression -->
:::{admonition} TODO
fill in a reasonable concrete use case of this repeating the tab structure filling in the variables
:::

This generally has to be written mathematically to be solved, but the implementation is then translated into a programming language. 

### A common problem to solve

Then the goal in creating the learning algorithm is to find the right details, if we take the mathematical representation above, we need to find the right $\theta$.  

Learning algorithms output that and then allow us to have a complete prediction algorithm.  

## There are different types of AI 

### Predictive AI saw success first 


### Generative AI is currently popular

These are generally {term}`LLMs <large language model>` so they are a {term}`model` of a natural or spoken language (or many of them, or a mix of natural and programming languages).  

More broadly a {term}`generative model` is a concept from statistics, this can be as simple as a model to predict the species of iris from the measurements of lenght and width of the petals and sepals. A generative model can be used to sample new data that looks like the training data.  





<!-- 
## Limitiations


-  -->

