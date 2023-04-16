# Programmer: E Ching Kho (He/him)
# Date: 12th February, 2023
# Claim: This program is for Queen's University Undergraduate Project purpose.
#        I respect the Terms of Service and will use a reasonable crawl rate
# Description: This program will extract useful information from Engadge Website and create a csv file for further uses

# import libaries
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
from csv import writer

# Create an instance of Chrome web driver
driver = webdriver.Chrome()

# Set the base URL of the search result
base_url = "https://search.engadget.com/search?p=ea+electronic+arts&ei=UTF-8&fr=engadget&fr2=sortBy&sort=date&cat=all&b=" # For EA
# base_url = "https://search.engadget.com/search?p=nintendo&ei=UTF-8&fr=engadget&fr2=sortBy&sort=date&b=" # For Nintendo

# Set the number of pages wanted to scrape
num_pages = 99

# Write the data into a csv
with open('../ea_engadget.csv', 'a+', encoding='utf8', newline='') as f: # f shortform for file, a+ append and create if file not exist
# with open('ntd_engadget.csv', 'a+', encoding='utf8', newline='') as f: # f shortform for file, a+ append and create if file not exist
    thewriter = writer(f)
    header = ['title', 'author', 'date', 'lure', 'link']
    thewriter.writerow(header)

    for i in range(num_pages):
        # construct the URL for the current page
        if i == 0:
            url = base_url + '1'
        else:
            url = base_url + str(i) + '1'

        # Navigate to the website url
        driver.get(url)

        # Since engadget is a dynamic website, wait for the website to load completely
        driver.implicitly_wait(1)

        # Get the page source and pass it to BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Content is under the ul element with class "compArticleList"
        content = soup.find('ul', class_='compArticleList')
        # print(content)

        # Only need each item content section within the ul element
        li_elements = content.find_all('li', class_='ov-a')

        # for item in li_elements:
        #     # print(item.text)
        #     # link = item.find('a', class_="thmb")['href']
        #     # print(link, end=", ")
        #     title = item.find('h4').text
        #     print(title, end=", ")
        #     lure = item.find('p', class_="fz-14").text
        #     print(lure, end=", ")
        #     author = item.find('span', class_="pr-15").text
        #     print(author, end=", ")
        #     date = item.find('span', class_="pl-15").text
        #     print(date, end="\n\n ")


        # Loop through each item to extract each item information
        # i = True # For testing purpose
        for item in li_elements:
            title = item.find('h4').text
            try: 
                author = item.find('span', class_="pr-15").text
                author = author[3:]
            except:
                author = "Unknown"
            date = item.find('span', class_="pl-15").text
            try:
                lure = item.find('p', class_="fz-14").text
            except:
                lure = "None"
            link = item.find('a', class_="fz-20")['href']

            info = [title, author, date, lure, link]
            thewriter.writerow(info)