# Report №1 (draft)

1) Describe the general problen about fakes info in our world.
2) Identify the businees problem or social value of this project.
3) Identify the ML problem.
4) Shortly describe the data that we need to this problem (in general, not related to specific dataset).
5) EDA.
6) Future plans of project workflow.


### 1. **General Problem: Fake Information in the World**
Fake news and misinformation have become rampant in today’s digital age. Social media platforms, like Twitter, often become breeding grounds for false information due to the speed at which content is shared and the lack of rigorous fact-checking. Misinformation can lead to public confusion, panic, and even dangerous situations, especially during disasters or emergencies. The challenge is distinguishing real, impactful events from posts that simply use emotionally charged language without any basis in actual events.

### 2. **Business Problem / Social Value**
The main social value of our project is to reduce panic and confusion by helping users, emergency responders, and organizations differentiate between real disaster news and emotional or sensationalized content. Businesses, particularly those in emergency services, news verification, or social media monitoring, can use this tool to quickly filter real news from noise. This could also aid in automating disaster response and resource allocation by ensuring the information acted upon is reliable.

### 3. **ML Problem**
The ML problem here is a **classification task**. The goal is to classify tweets into two categories: 
- **True disaster news** 
- **Emotionally charged, non-news content** 

This is a text classification problem where our model will learn from patterns in the text to determine whether a tweet refers to an actual disaster or just emotionally laden content.

### 4. **Data Requirements**
For this problem, you will need a dataset that includes:
- Tweets or other short-form texts with information related to disasters (such as natural disasters, accidents, or other emergencies).
- Each tweet should be labeled as either:
  1. **Real disaster news**
  2. **Non-news emotional tweets** 

Additionally, having metadata such as time of posting, user information, and location might be helpful to identify trends and patterns related to real events. For example, real disaster tweets often cluster around certain times and are reported by users in affected regions.

### 5. **Exploratory Data Analysis (EDA)**
Our EDA should focus on:
- **Text analysis**: 
  - Analyzing word frequency in real vs. emotional tweets.
  - Common phrases, hashtags, or words used in each category.
  - Sentiment analysis to see if emotional tweets tend to have extreme positive or negative sentiments compared to factual tweets.
  
- **Time-based analysis**:
  - Look for trends in when tweets are posted. Real disaster news might surge during an actual event, while emotional tweets may be more spread out.

- **User behavior**: 
  - Are there certain users who tend to tweet more real news vs. emotional content?

### 6. **Future Project Workflow**
- **Week 1-2**: Data collection and labeling (if you're working with an unlabeled dataset).
- **Week 2-3**: Exploratory data analysis and feature engineering. This involves cleaning the data, tokenizing the tweets, and generating potential features like tweet length, sentiment, and keyword presence.
- **Week 3-4**: Model development and training. Start by trying various machine learning models like logistic regression, decision trees, and deep learning models like LSTMs or Transformers for NLP tasks.
- **Week 4-5**: Model evaluation and optimization, focusing on improving accuracy and reducing false positives.
- **Week 6**: Deployment and real-time testing, possibly integrating your model into a real-time Twitter feed to monitor its performance.
