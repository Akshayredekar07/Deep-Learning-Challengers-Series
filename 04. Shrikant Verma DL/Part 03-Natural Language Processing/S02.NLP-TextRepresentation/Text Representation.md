**Text Representation**

**Finding similar Medium articles**

You are working as a Data Scientist at Medium.  
Medium is an online publishing platform which hosts a hybrid collection of blog posts from both amateur and professional people and publications.  
In 2020, about 47,000 articles were published daily on the platform and it had about 200M visitors every month.  

**Problem Statement:**  

You want to give readers a better reading experience at Medium. To do that, you want to recommend articles to the user on the basis of the current article that the user is reading.  
More concretely, given a Medium article, find a set of similar articles.  

**How would a human find similar articles in a corpus?**  

1. Look at the title - find similar titles.  
2. Find articles by the same author.  
3. Go through the text, understand it, and group the articles within broader topics.  

**Let's have a look at the data**  

**What data are we going to use?**  

Each article in Medium has a title, article text, and author associated with it. To begin with, this data should be sufficient to understand the articles and to find similar articles.  
The user might like articles which belong to the same topic -  
For example: if a user is reading an article on Neural Networks, he/she might be interested in a similar article which talks about Convolutional Neural Networks (both of these articles belong to a broader domain of Deep Learning).  

**How do we get this data?**  

Well, we can scrape it from Medium (scraping is covered later in the track).  
We’ve done that already.  
We have a collection of Medium articles with its title, subtitle, author, reading time, text, and the link of that article.  
