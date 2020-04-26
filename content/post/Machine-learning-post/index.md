---
authors:
- admin
categories:
- Machine Learning
date: "2016-04-20T00:00:00Z"
draft: false
featured: false
image:
  caption: 
  focal_point: ""
  placement: 2
  preview_only: false
lastmod: "2019-04-17T00:00:00Z"
projects: []
subtitle: 'The basics of Machine Learning explained in an easy to understand format :brain:'
summary: Machine Learning explained without the jargon.
tags:
- Machine Learning
- Artificial Intelligence
title: 'What Is Machine Learning?'
---

Hello everyone, over the past couple of years you have probably heard about the new phase within the industry regarding Artificial Intelligence. As a result, many companies have gone on to hire data specialists & machine learning engineers without truly understanding what they provide and whether their company has the right infrastructure set up. In light of this, I will be talking about some of the key concepts in Machine Learning which is a subset of artificial intelligence, and is the “Field of programming which gives computers the ability to learn without being explicitly programmed.” Arthur Samuel

What does that mean exactly? Well from a very high-level (easy-to-understand) perspective it means that data is input into a model created by data scientists, which from the existing data will use the data given to the computer and make a prediction. A good example is predicting stocks, you provide data of existing stock information for the program to make a prediction of stock prices.

There are three main types of Machine Learning that I wish to talk about within this post, where I will go into a little bit of detail regarding how they work, for people unfamiliar with these concepts and for those wishing to refresh their memory.

**Reinforcement Learning**

Reinforcement learning is commonly used within fields such as Robotics & Game Theory and where there is a multitude of actions that can occur, given an initial state. What this means, is that the machine will learn by itself, then optimize from the previous choices made to take the next best action. This is different from the other types of machine learning which may not output the “optimised solution”. An example of this would be if you were to have a Flappy Bird game programmed by reinforcement learning, the first attempt of the bird may initially collide to the ground, then it may collide with a pole, however, reinforcement learning will eventually optimise the birds route after each failed attempt until it creates the best path for the bird, to get as far as possible in the game. This could be applied to other types of games as well, for example, an AI from a chess game, can be programmed using reinforcement learning, and from each game played it will gradually become better over time. Reinforcement learning is generally not used within the industry apart from the fields mentioned above, mainly because companies prefer to use other methods which are generally easier to program, however noticeable exceptions that are becoming increasingly popular within the industry are Chatbots and devices such as Alexa and Siri, these are examples of reinforcement learning.


**Supervised Machine Learning**

Supervised Machine Learning, is the second of the three fields within Machine Learning and by far the most popular field used within the industry. This is because Supervised Machine Learning is trained on labelled data and there is already a rough idea of what the prediction will be. For example, what mode of transport will Person A take to work? Or what is the value of Company A? Thus, before the algorithm is implemented, there is already a rough idea of what algorithm will predict. Hence, the program knows there is a value/category which the outcome can be picked from. One of the most common types of Supervised Machine Learning examples within the Data Science community is the Titanic: Machine Learning problem, where you use Classification to predict whether each passenger survives or doesn’t. I will now try and break it down further.

**There are two main types of Supervised Machine Learning Algorithms:**

* Regression
* Classification


Classification is used in problems such as predicting what mode of transport Person A takes to work, so the training data (data which the model uses to learn from) is based on categorical data. Hence, as the classes of transport are already defined, i.e. bus, car, train etc. the question is a classification problem.

Whereas Regression is used in problems which still have a rough idea of what value will be predicted, however, the input data is continuous and an example would be predicting the weight of Person A or the value of Company A. Hence, regression is used when you have a continuous output.

This can be further broken down however, for now, I will leave it at that and I hope to break it down further in a later post.

**Unsupervised Machine Learning**

Finally, the last type of Machine Learning, where you have your training data but no idea what the prediction will be based on the input data. You may be thinking, that sounds awfully similar to Reinforcement Learning? However, reinforcement learning is focused on taking the next action, i.e. simulating the next move within a Chess game. Whereas, Unsupervised Learning is more focused on finding the patterns and differences between sets of data. An example of this would be a 3D Cluster with the respective axis of age, salary and gender and thus, create its own categories for data points which have similar results on the 3D cluster.


Unsupervised Learning is different from supervised due to assigning what data points it believes would be categories rather than having pre-defined categories such as mode of transport for Person A to work.

Unsupervised learning will generally require more “computational” power than supervised learning, and supervised learning is generally more accurate than unsupervised learning. So where possible, it is advised to use supervised learning over unsupervised learning.

I will delve into more detail about unsupervised learning in a different post, as I wish to keep this article about the basics, and there is so much more to explore.

Thank you for taking the time to read my “What Is Machine Learning.” I hope you found it useful, regardless of your industry & sector of work/education and now you won’t need to nod your head when a colleague mentions Artificial Intelligence or Machine Learning without knowing what it truly achieves within the workplace.