Capstone Project Final Proposal

AirBnb Review Analyzer by Discovering Latent Rating Aspects


Description
1) Motivation:
It is easily to get overwhelmed by the number and the diversity of online product reviews. In particular, when we quickly glance through overall rating scores and reviews, it's still difficult for us to find out each user's opinion towards certain product aspects. Thus I want to develop a review analyzer that can help us easily digest and exploit product reviews in understanding each reviewer's opinions at a topic level hidden in the review corpus.

2) Goal:
Build an AirBnb review analyzer (presented by a webapp), which shows rated aspect summarization and user preferences for high/low reviews, and ultimately(if time permits) a query-based/personalized recommender system.


Techniques
(Details will be added later in separate files)

1) Latent Aspect Rating Analysis (LARA):
Given a set of reviews with overall ratings, LARA aims at analyzing opinions expressed in each review at the level of topical aspects to discover each individual reviewer’s latent rating on each aspect as well as the relative importance weight on different aspects when forming the overall judgment.
The overall approach consists of two stages. First we perform Aspect Segmentation in a review document based on the given keywords describing aspects to obtain text segment(s) for each aspect.
Then based on the aspect segmentation results in each review, we apply a novel Latent Rating Regression(LRR) model to analyze both aspect ratings and aspect weights. (The LRR model assumes that we know which words are discussing which aspects in a review, so Aspect segmentation comes first.)

2) Topic Modeling:
   a) Non Negative Matrix Factorization(NMF):
   NMF approximates a nonnegative matrix by the product of two low-rank nonnegative matrices. Since it gives semantically meaningful result that is easily interpretable in clustering applications, NMF has been widely used as a clustering method especially for document data, and as a topic modeling method.
   b) Latent Dirichlet Allocation (LDA):
   LDA is a way of automatically discovering topics that a text document contains. In more detail, LDA represents documents as mixtures of topics that spit out words with certain probabilities.
   In Natural Language Processing, LDA is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar.

3) A query-based Recommender System as a product from LARA, which takes into account one reviewer's preference and opinions towards different aspects.


Challenges
1) Data:
In the existing dataset, one Airbnb listing usually has less than 5 reviews. This limitation will probably affect model's performance since it can get highly biased in terms of each rating aspect and their weights, especially when there is only one or two polarized reviews.
Due to limited time, there won't be more scraped data to complement existing reviews; But the model will learn from the whole review corpus using both supervised and unsupervised algorithm, which can mitigate the bias mentioned above.

2) Latent aspect rating analysis:
A major challenge in solving the problem of LARA is that we do not have detailed supervision about the latent rating on each aspect even though we are given a few keywords describing the aspects. Another challenge is that it is unclear how we can discover the relative weight placed by a reviewer on each aspect.

3) Recommender system:
There are few research articles available in building a Recommender System as a product of LARA. Thus it will be less directed and more uncertain from LARA to a personalized recommender system.
Plus, the review matrix will be extremely sparse and have lots of cold starts.

4) Others coming soon...


Timeframe
1) Week-1:
Get data; Data cleaning and integration;
Exploratory Data Analysis(EDA);
Finish final project proposal: define project scope and knowing where the limits can be;
Learn and explore techniques used in the project, especially LARA and LDA;
Build models and pipeline;
Have a baseline model and at least an improved one, so that I will know the right improvement direction.

2) Week-2:
Model improvement; Select final model;
Data visualization on model outputs and qualitative inferences;
Build a webapp on flask to present the review analyzer and project's results;
Draft a presentation PPT: if there is no time for a well-designed functioning webapp, PPT will be the main source to show my project.
Code cleaning and Repo organization.


Data
1) Source: AirBnb dataset, http://insideairbnb.com/get-the-data.html
2) Info:
   3.81 GB; CSV files;
   Including 43 cities' AirBnb listings(1.55 GB) and reviews(2.26 GB) in recent years
4) Sample data:
   Please see the two csv files in the data folder; They are 10 sample listings and 10 sample reviews.
