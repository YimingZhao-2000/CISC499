# Programmer: E Ching Kho (He/him)
# Date: 18th January, 2023
# Claim: This program is for Queen's University Undergraduate Project purpose.
#        I respect the Terms of Service and will use a reasonable crawl rate
# Description: This program will extract useful information from Website and create a csv file for further uses

from bs4 import BeautifulSoup
import requests
from csv import writer

# url = "https://www.businesstoday.in/latest/economy"
url = "https://www.wsj.com/market-data/quotes/EA?mod=searchresults_companyquotes"

page = requests.get(url) # request a GET url

# print(page) # Should be <Response [200]> which is success

# Getting the content of the page with html format
soup = BeautifulSoup(page.content, 'html.parser')

# Only need each item content section
lists = soup.find_all('ul', class_="WSJTheme--cr_newsSummary--2RNDoLB9 ")

# Write the data into a csv
with open('economyNews.csv', 'w', encoding='utf8', newline='') as f: # f shortform for file
    thewriter = writer(f)
    header = ['title', 'link', 'date', 'content']
    thewriter.writerow(header)

    # Loop through each item to extract each item information
    # i = True # For testing purpose
    for list in lists:
        heading = list.find('a', href=True)
        title = heading.text
        link = heading['href']
        date = list.find('span').text
        content = list.find('p').text

        info = [title, link, date, content]
        thewriter.writerow(info)

        # if i: # For testing purpose
        #     # print(title)
        #     # print(link)
        #     # print(date)
        #     # print(content)
        #     print(info)
        #     i = False
