# CISC499
The Queen's undergraduate project

### Description: 
This project will mainly focus on the price prediction of one selected stock and try to improve model accuracy. Then we will use XAI tools and techniques to analyze the reliability of our model and build explainability features to explain model predictions by visualization. If time permits, we will write a quantitative trading algorithm, practice investing based on our model, and improve the model by other machine learning methods.  

### To Run the code
In terminal,
1. `pipenv shell` to activate the shell in the virtual environment
2. `pipenv run python -m ipykernel install --user --name=CISC499` Ipykernel makes the kernel for use in Jupyter
3. `pipenv run jupyter notebook` to start up jupyter notebook and run all the ipynb code in the directory

### Objective: 
>- Design a model with stable high accuracy
>- Present explainability features using XAI tools
>- Implement trading algorithm and practice investing; expect a positive return

### Field of Interest:
- Game industry
    >- Have domain knowledge
    >- It will be affected both by stock market.
    >- Their consumers are one of the group who are the most likely to evaluate their products online.

### The degree of evaluation:
    - The history stock price : using time series model
    - How well the company health
    - The macro economy
    - New and press by the company
    - User and IGN 's comments and score for their products.


## Game Companies we are analyizing on
### Nintendo Co. Ltd. ADR
#### Information Overview
- INDUSTRY  : Toys & Games
- SECTOR : Consumer Goods
- Data type 
    >- Table
    >- Articles
- Source of Data
    >- [Markect Watch](https://www.marketwatch.com/investing/stock/ntdoy/company-profile?mod=mw_quote_tab)
    >- [IGN Score for Wii](https://www.ign.com/reviews/games/wii)
    >- [IGN Score for Wii U](https://www.ign.com/reviews/games/wii-u)
    >- [Stock Hitorical Data]
    >- Analyst Research

###  Elecreonic Arts Inc. Common Stock
#### Information Overview
- Nasdaq Listed - Nasdaq 100 
- Consumer Discretionary
- Data type 
    >- Table
    >- Articles
- Source of Data
    >- [Nasdaq Historical Data](https://www.nasdaq.com/market-activity/stocks/ea/historical)
    >- [News](https://www.nasdaq.com/market-activity/stocks/ea/news-headlines)
    >- [Press release](https://www.nasdaq.com/market-activity/stocks/ea/press-releases)
    >- [YChart](https://ycharts.com/companies/EA/pe_ratio)
    >- Analyst Research