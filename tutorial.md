(what-is-ai)=
# What *is* AI? 

Goals:
- define AI in multiple ways and contextualize the term usage
- describe *how* modern AI is built and how that differs from older algorithms


::::::::{important}

the widespread Generative AI tools that we currently have break two key assumptions that have been true about technology previously built:

:::::::{tab-set}
:::{tab} Traditional Software
- Computers reliably produce correct information 
- A person designed every behavior of the application and tested it completely, for all possible inputs to make sure it is correct
:::
:::{tab} Generative AI 
- Generates words[^beyondwords] randomly, conditioned on the input
- People design machine learning algorithms that write the algorithms that go into the world. ML is tested statistically on a random sample of tests.
:::
:::::::
:::::::::


::::::::::::::{attention} Reader note

Some sets of figures are included in tabs

:::::::{tab-set}
::::{tab} A
:sync: start
so that an image
::::

::::{tab} B

so that an image can be built up 
::::

::::{tab} C
so that an image can be built up  over multiple
::::

::::{tab} D
:sync: end 

so that an image can be built up  over multiple steps to a final, more complex image
::::
::::::::::

If you prefer to see the complete image while reading without going throug the tabs, leave the tabs above on the last step, the ones below will be on that last step too.  

If you wan to see the first image first below, select the A tab above. 


::::::::::::::

## AI *is* computer science

Building {term}`artificial intelligence` has been the goal of computing research (computing is a  broader feild including both computer *science* or computer *engineering*) the whole time.  Computing is not a very old discipline, but can trace its roots to math and electrical engineering (and through that, physics).  At some institutions, computer science and computer engineering are in the same department and at others they may be separate. Computer Science has especially been focused on developing AI  as a goal. 

:::::::{figure} assets/cs-ai-venn.svg
:label: csaivenn

A venn diagram showing CS as a broad feild with AI as a sub feild
::::::::::

AI is a field of study, which means there is a community of people egaged in this work. 

{term}`AI <artificial intelligence>` can be done in many ways and has been over time, but the current dominant paradigm is machine learning.  
The formal definitions of AI are broad and in that way, almost everything in a computer can be considered to "be AI" but, in this moment, AI typically refers to a few things:
- {term}`Large Language Models <large language model>` (and multimodal mdoels with the same basic goal) that produce text and accomplish high level goals based on english
- {term}`predictive models <predictive model>` that predict specific things or automatically label things and are the result of machine learning
- complex systems that combine the two things above with additional logic



:::::::{figure} assets/cs-many-venn.svg
:label: csmanyvenn

A venn diagram showing CS as a broad feild with many subfeilds
::::::::::

What is common across all of these, and all things in computer science, is {term}`algorithms <algorithm>`. Algorithms have gotten a lot more attention recently, but they are not fundamentally new, mathematicians have developed and studied them formally for centuries and informally, people have developed them all over. 



:::{figure} assets/cs-algo.svg
:label: algocenter

Algorithms are at the center and what is common in all of CS. 
:::




## How Algorithms are Made 

A familiar way to think about what an {term}`algorithm` is as a recipe. A recipe consists of a set of ingredients (inputs) and a set of instructions to follow (procedures) to produce a specific dish (output). Computer algorithms describe the procedure to produce an output given an input in order to solve a problem. 

Mathematicians have developed algorithms for centuries by:
1. Selecting a problem to solve in the world
2. Representing for the relevant part of the world mathematically
3. Working to solve the problem in the mathematical space
4. Documenting their process so that it is repeatable


:::::::{tab-set}
:::::{tab} Begin
:::{figure} assets/algodev-1.svg
:label: algodev-people
:sync: start

Algorithms are *developed* by people, not naturally occuring things that we observe or *discover*
:::
:::::

::::::{tab} Select
:::{figure} assets/algodev-2.svg
:label: algodev-select

The person picks a problem to solve and some part of the world as a context for that solution-whether they attend to the details of this choice carefully or not
:::
::::::

::::::{tab} Decisions
:::{figure} assets/algodev-3.svg
:label: algodev-choose

A person thinks about the problem and makes *choices* about each step
:::
::::::

::::::{tab} Approximate
:::{figure} assets/algodev-4.svg
:label: algodev-approx

In general, we cannot write math that exactly describes the world so we pick some way to simplify the part of the world that we wish to study, that we think is relevant. 
:::
::::::


::::::{tab} Represent
:::{figure} assets/algodev-5.svg
:label: algodev-math

As they make simplifications, they write down a mathematical representation of the simplification
:::
::::::

::::::{tab} Solve
:::{figure} assets/algodev-6.svg
:label: algodev-solve

Once it is represented, they can solve it, applying mathematical techniques
:::
::::::

::::::{tab} Distill
:::{figure} assets/algodev-7.svg
:label: algodev-spokenalgo
:sync: end 

Finally, the steps to re-create the solution for a similar problem is written down so that other people can follow the steps, or apply the algorithm
:::
::::::


:::::::::::::


Computer Scientists do the same, with the main change being that the solution is expressed in a {term}`programming language` instead of in a spoken language, for example `Python` instead of English. 


:::::::{tab-set}
:::::{tab} Simple changes
:::{figure} assets/algodev-8.svg
:label: algodev-cs
:sync: start

Computer Scientists might use different tools to develop algorithms or terms to document things, but the process is mostly the same
:::
:::::


::::::{tab} Implementation 
:::{figure} assets/algodev-9.svg
:label: algodev-plalgo
:sync: end 

The main difference is the final algorithm is written in a programming language for a computer to execute instead of a person following the steps. 
:::
::::::


:::::::::::::

The challenge with this is as we approach more complex problems as our goal to try to delegate to a computer, the approximaitons we make get in the way more. Writing an algorithm to add numbers together or find an exact match for an item from a list is straightforward, writing an algorithm to detect if a set of pixels represents a person or not (e.g. if there is a person in front of a self driving car) is much more complex. 

The traditional way of developing algorithms works well for problems where we have a good mathematical repesentation of the part of the world we need to compute and people can describe the steps that need to occur in terms of calculations a computer can carry out. 

This is where {term}`machine learning` comes in. 




## What is ML? 
:::{important}
Current success in AI is based on {term}`machine learning`
:::
In {term}`machine learning`, we change the process a little.  

Instead of solving the problem and figuring out precise steps to carry out, the people define a generic strategy, collect a lot of examples, and write a {term}`learning algorithm` to fill in the details of the generic strategy from the examples. 
Learning algorithms are developed the way we have always developed algorithms, but then these algorithms essentially write a {term}`prediction <prediction algorithm>` or inference algorithm that is what gets sent to the world. 


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

(mlassumption)=
### A common assumption


All ML has some sort of underlying assumptions, almost all ML relies on two key assuptions, that can be written in many ways: 



::::{tab-set}
:::{tab-item} Plain English
:sync: english 
1. A relationship exists to that we can determine, or predict `outcome or target`  from `input`. 
2. Given enough examples a computer can find that relationship

where: 
- `outcome or target` is the goal of the task 
- `input` is the information to be used to predict that target
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

Given a row index matrix `training_in` and a vector `training_out`  we can write: 
```Python
data = [(in_i,out_i) for in_i, out_i in zip(training_in, training_out)]
parameters = learning_algo (data)
predictor = lambda input_example: pred_algo(input_example, parameters)
pred_output = predictor(test_input)
```

where[^whypy]: 
- `pred_algo` is a function template that uses in `parameters` to customize the calculation 
- `parameters` can change how `pred_algo` works to adapt it to different contexts, the {term}`lambda` keyword makes a new function that takes in only the input 
- `predictor` take on input sample and computes the predicted output


and for any valid `test_input` we will get a valid `pred_output`


:::
:::{tab-item} Diagram
:sync: end

(mlassumptiondiagram)=
```{mermaid}
flowchart LR
    x(input <br>x) -->  f[f ] --> y(output <br>y)
```
<!--  <sub>#952;</sub -->

:::

::::

[^whypy]: Python is a programming language specifically designed for [readability](#pythonreadability).

This, alone, is not that different than the traditional way of developing algorithm, we have to assume a way to get from some input to the desired output exists for that to happen.  However, in machine learning this is a bit more specific, we assume that there is a specific $x$ and $y$ that are available to us[^unsup] and that from the $x$, we can {term}`compute` a value for $y$. 

:::::{important}
Computers[^quantum] can **only** do a limited set of mathematical operations: 
- add (a+b)
- multiply (a*b)
- negate (-a)
- invert (1/a)
- compare (a< b)

and store and retreive values represented in binary.

Everything else is done by combining these operations 
::::::::

[^unsup]: there is also unupservised or semi-supervised where the $y$ is either unknown or only available for some samples, but they still assume that it exists and the $x$ can be used to *compute* it. 

[^quantum]: quantum computers, which are not yet available for consumer use or even broad research use, represent data with probabilistic qubits instead of traditional binary


To make this concrete, this could be as simple as a linear regression


::::{tab-set}
:::{tab-item} Plain English
:sync: english 

We can predict the tip for a restaurant bill based on the total bill, by multiplying by some percentage and adding a flat amount. We can determine the percentage and the amount to add from previous bills. 

:::
:::{tab-item} Math
:sync: math

Assume have vectors $x$ and $y$

- $y_i = f(x_i) = mx_i+b$
- $\theta = [m b]$
- Ordinary least squares can sovle this or any minimization algorithm can solve:

```{math}
:label: assumptionmath
\theta= \mathcal{A}(x,y)= \arg_\theta\min ||y - mx+b||_2^2
```

or, equivalently, element-wise


```{math}
:label: assumptionmathelement
\theta = \mathcal{A}(x,y) = \arg_{m,b}\min \sum_{i=1}^D (y_i - mx_i + b)^2 
```

where:
- $x$ is an input or {term}`features <feature>`
- $y$ is an output, {term}`label`, or {term}`target`
- $\mathcal{A}$ is an algorithm


:::
:::{tab-item} Code (Python)
:sync: python

Given a row index matrix `training_in` and a vector `training_out`  we can write: 
```Python
def learning_algo(data):
    theta0 = initialize_theta()
    abs_pred_error_i = lambda t,x,y: abs(y - pred_algo(x,t))
    total_pred_error = lambda th: sum([pred_error_i(th,x,y)] for x,y in data)
    # optmize so that the error is minimized
    theta = minimize(total_pred_error,theta0)
    return theta


def pred_algo(x,theta):
    m,b = theta
    return m*x + b

data = [(in_i,out_i) for in_i, out_i in zip(training_in, training_out)]
parameters = learning_algo (data)
predictor = lambda input_example: pred_algo(input_example, parameters)
pred_output = predictor(test_input)
```

where[^whypy]: 
- `parameters` can change how `pred_algo` works to adapt it to different contexts, the {term}`lambda` keyword makes a new function that takes in only the input 
- `predictor` take on input sample and computes the predicted output
- `minimize` takes a function and parameters for it and finds values of the parameters to the function that get the smallest possible value. 

and for any valid `test_input` we will get a valid `pred_output`


:::
<!-- :::{tab-item} Plot
:sync: end



::: -->

::::

This generally has to be written mathematically to be solved, then the implementation is then translated into a programming language for a computer to execute.




### A common problem to solve

Then the goal in creating the learning algorithm is to find the right details, if we take the [mathematical representation](#assumptionmath) above, we need to find the right $\theta$.  

Learning algorithms output that and then allow us to have a complete prediction algorithm.  


A learning algorithm and prediction algorithm are linked by a shared {term}`model`. The prediction algorithm is basically the model treated as a template so that once the parameters are set it becomes a simple input output function.  The learning algorithm is what people work on how to write how to find the right parameters to make predictions in a specific domain. 

## ML is classified in many ways

AI can be classified by how it is developed:
- traditional methods (rule based systems,etc)
- Machine learning
- hybrid systems that combine multiple types (agentic AI)

Most current things are ML, and the underlying assumptions come in different forms.  

ML can be classified in many different ways too:
- when we focus on the *learning* problem, we classify into {term}`supervised <supervised learning>` and {term}`unsupervised <unsupervised learning>` learning based on availability of the {term}`target` variable and the *type* of prediction we want to make discrete ({term}`classification` or {term}`clustering`) or continuous ({term}`regression`)
- if we focus on *what* is learned to make decisions we can classify into {term}`discriminative <discriminative model>` or {term}`generative <generative model>`. 
- if we focus on the specific assumptions, we can classify by the {term}`model class`

We can describe a model with each of these descriptors for example:
- ChatGPT, Gemini, and Claude are examples of {term}`large language model`s and specifically {term}`GPTs <generative pretrained transformer>` which are a type of {term}`generative model` implemented with {term}`deep learning` they are trained with  {term}`unsupervised <unsupervised learning>` initially followed by `supervised <supervised learning>`
- the original HALO player ranking algorithm was also a {term}`generative model` but it was primarily used to make predictions of what would be a good matchup, rather than generating new sequences of win/loss for opponent pairs. it was trained with a `supervised <supervised learning>` approach of past player matchups. 


<!-- 
### Discriminative vs Generative


These are generally {term}`LLMs <large language model>` so they are a {term}`model` of a natural or spoken language (or many of them, or a mix of natural and programming languages).  

More broadly a {term}`generative model` is a concept from statistics, this can be as simple as a model to predict the species of iris from the measurements of lenght and width of the petals and sepals. A generative model can be used to sample new data that looks like the training data.  
 -->

<!-- 
## Limitiations


-  -->

## What is an LLM? 

While AI has been a research area in computing since the beginning of computing, AI came into most common use when ChatGPT was released. ChatGPT is chatbot interfact to the GPT family of  {term}`LLMs <large language model>`. This and large scale models of vision for image generation or audio for audio production, etc all work on the same basic idea.  For LLMs, specifically this is:

- **model** is a simplification of some part of the world
- **language** is a tool for communicating consisting of words and rules about how to combine them
- **large** refers to the number of {term}`parameters <parameter>` is big

Specifically, they model language by using a lot of examples and a statistical model. In math:  

```{math}
:label:condmodel
P(w_j| w_{j-1}, w_{j-1},\ldots, w_{j-c})
```

where $c$ is callend the context window.  

In English, this says that the model represents a {term}`proabability distribution` of possible next words($w_j$) given a past sequence of $c$ words. 


[](#condmodel) is *implemented* in a computer using neural network.  A Neural networks is computationl model for approximating a function defined by a number of [artificial neurons](#sec:nndef). Neural networks approximate complex functions by combining a lot of simple functions together.  



