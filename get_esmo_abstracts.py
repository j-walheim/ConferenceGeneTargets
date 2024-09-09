from bs4 import BeautifulSoup
import csv

# Read the HTML file
with open('data/Posters - ESMO Congress 2024.html', 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

# Find all presentation cards
presentation_cards = soup.find_all('div', class_='card presentation')

# Prepare CSV file
with open('abstracts_detailed.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Abstract Number', 'Title', 'Authors', 'Presentation Date', 'Topic', 'Abstract']
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csv_writer.writeheader()

    # Extract information from each presentation card
    for card in presentation_cards:
        abstract_number = card.find('div', class_='property-container internal p').find('div', class_='property').text.strip()
        title = card.find('h4', class_='card-title').text.strip()
        authors = card.find('ul', class_='persons').text.strip()
        presentation_date = card.find('div', class_='property-container cslide text p').find('div', class_='property property-text').text.strip()
        topic = card.find('span', class_='label p a-pt a-pt-hover').text.strip()
        
        abstract_div = card.find('div', class_='abstract')
        abstract_text = abstract_div.get_text(strip=True, separator=' ') if abstract_div else ''

        csv_writer.writerow({
            'Abstract Number': abstract_number,
            'Title': title,
            'Authors': authors,
            'Presentation Date': presentation_date,
            'Topic': topic,
            'Abstract': abstract_text
        })

print("CSV file 'abstracts_detailed.csv' has been created successfully.")
