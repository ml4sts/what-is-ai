
# AI Risks and Ethical considerations

::::{important}

This page represents a *small* cross section of known and potential impacts of AI, with an emphasis on deployed systems and generative models. 

For each point, there are scholarly sources linked supporting the point and in most sections the introduction includes links to trusted sources aimed at more general audiences.  
::::


The most important thing to know is that these technologies reflect how they are built.  Machine learning fundamentally uses the past to make decisions or generate content without good ability to use contextto determine what was bad in the past that we might not want to keep for the future.  



## Discrimination and Bias

In general, technological systems reflect the values of the people who build them.  This may be explicit or implicit.  With machine learning, the way the values of the builders are expressed is more distant from what goes out into the world; it is often less carefully checked. 


## Evironmental Impact

Training large language models requires an immense amount of energy and water (for cooling), using them requires less per use, but still a lot, especially when aggregated across all of the iterative prompting across all of the users. 
[HuggingFace :hugging_face: ](https://huggingface.co/), a company that provides a platform for sharing trained models, datasets, and applications provides a policy primer that is a good starting point @ai_environment_primer. Hugging Face researcher Sasha Luccioni also participated in an [NPR Podcast on Climate impact of AI](https://www.npr.org/transcripts/1250261120).

Some key highlights: 
- {cite:t}`strubell2020Energy` found that training an {term}`LLM <large language model>` with only 213 million parameters [^gpt3] was responsible for 625,155 points of $CO_2$ roughly equivalent to the life time emissions of five cars.
- {cite:t}`luccioni2023estimating` followed a sample model, of similar size to GPT3 (176 Billion parameters vs 175) through its full lifespan from training to use. They found that the total carbon impact of such a model is more than training alone. 
- @luccioni2024power break down model energy use by different types of tasks (text classificaiton, summarization, text generation, image generation) and found that image generation uses nearly 60 times more energy than any text based task and has a carbon impact comparable to driving the average passenger vehicle about a mile[^car], using about as much energy as charging a cellphone half way [^charge]
- in addition to the electriciy use, large data centers use tremendous amounts of fresh water and those numbers have increased rapidly recently ( google increased by 20% and Microsoft by 34% from 2021 to 2022) @ren2023water
- Depending on the location of data center that handles the query, GPT3 could consume[^consumedef]  7.2ml (Netherlands) to 48.3ml (Washington, US)  of water per query meaning a full 500ml (16.9oz) bottle of water is consumed for every 11 (Washington) to 70 queries @ren2023water 

[^car]: @luccioni2024power found a range aroud of carbon impact in he 150-500g range across 1000 tests; the US EPA says that [the average passenger vehicle emits 400g per mile](https://www.epa.gov/greenvehicles/greenhouse-gas-emissions-typical-passenger-vehicle#:~:text=The%20average%20passenger%20vehicle%20emits%20about%20400%20grams%20of%20CO2%20per%20mile.)

[^charge]: The US EPA changed its estimate of cell phone energy use to 0.22kWh in January 2024 from 0.012KWh prior to that, an initial version of the @luccioni2024power study said equal to a full cell phone charge, but the publised version says half, accordingly. 

[^consumedef]: @ren2023water differentiates between water *withdrawal*, which includes all water taken temporarily or permanenly and water *consumption*, calculated as withdrawal minus discharge. They then focus on consumption because it reflects the impact of water us on downstream water availability. 

::::{margin}
:::{tip}
For more context on these numbers, try out a carbon footprint calculator such as one of the following provided by different organizations:
- [the nature conservancy](https://www.nature.org/en-us/get-involved/how-to-help/carbon-footprint-calculator/)
- [US EPA](https://www.epa.gov/ghgemissions/carbon-footprint-calculator) (they also provide a downloadable spreadsheet version)
:::

:::{note}
If you work with AI as a tool in your programming, you can use [codecarbon](https://codecarbon.io/) to measure the impact of your work
:::
::::

[^gpt3]: GPT3 (initial ChatGPT was GPT3.5) was 175 billion parameters @brown2020Language, nearly 1000x times as many as the model evaluated by  @strubell2020Energy 

## Privacy
Models can reveal the data they were trained on, which means that any user could see any value that the model builder put in to learn from. For generative models this can happen in normal use, and for predictive models, certain types of explanations, which are required by law in some contexts can be used to figure out the training data. 


- Researchers demonstrated an efficient attack on the {term}`production` chatGPT @nasr2023Extracting. Combined with the fact that free tools almost always save the data you provide them to train and improve their products, that means that anything you send to a chatbot, it could spit back out to another user. 
- IP risks. Content generated by an AI cannot be copyrighted, it is not ownable. 


## Impacts on Human Thinking

- A team at Univeristy of Toronto released a pre-print demonstrating a decrease in human creativity when asked to perform a task independently after exposure trials using an LLM @kumar2024human. This study measured two types of creativity using standard measures for each.
- A team at Northwestern's Kellog School of Business released a pre-print showing that LLM creativity is similar to human creativity, but when prompted to respond as female, old, or Black, they score considerably worse on creativity @wangpreliminary. This study measured one type of creativity. 
- A large-scale meta analysis found that AI+ human performance is often worse than the best of  human alone or AI alone. They also found that AI improved a person's performance mostly for *generation* tasks, but not for decision making tasks @vaccaro2024combinations.


(errorproof)=
## Errors[^hallucination] are inevitable

LLMs will make mistakes and give false information at some rate, guaranteed. These errors are because the LLM has no grounding in facts, it is generating text based on a probabilisitic model of language.  With a lot of feedback loops[^reasoning] or careful prompting good answers can be produced at high rates in some domains, but fundamentally they

A team at OpenAI and Georgia Tech showed that the training procedure means that LLMs are going to, with high probability, give a wrong answer over saying "I don't know"@https://doi.org/10.48550/arXiv.2509.04664. 

[^hallucination]: LLM errors are sometimes called hallucinations. I prefer to refer to them as errors. Hallucinations in a human brain occur when the brain stops using sensory information and treats its predictions about the world as if they are true and then without sensory information, the predictions become less and less related to the real environment the person is in. LLM hallucinations are defined by its output being incorrect. 

[^reasoning]: Reasoning models work, in broad terms, by adding extra text to the prompt and taking the output and then re-prompting. For example if your prompt is "what is 3+5" the reasoning model might actually prompt the LLM with something like, "make a plan to respond to: what is 3+5" and then it takes the models plan for answering the question  and prompts the model again to actually get an answer. You can think of it like prompting the model with specific prompts that help first write a better prompt for the thing you actually want and then actually using the better prompt. 