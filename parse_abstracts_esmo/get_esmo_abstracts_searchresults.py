from bs4 import BeautifulSoup
import csv

fname = 'data/ESMO_nonposters.html'
# Read the HTML file
with open(fname, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

# Find all presentation cards
presentation_cards = soup.find_all('div', class_='card presentation')

# Prepare CSV file
with open('data/abstracts_talks.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Abstract Number', 'Title', 'Authors', 'Presentation Date', 'Topic', 'Abstract']
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csv_writer.writeheader()

    # Extract information from each presentation card
    for card in presentation_cards:
        
        # Find the presentation number
        number_container = card.find("div", class_="property-container")
        number = "N/A"
        if number_container:
            info_div = number_container.find("div", class_="info", string="Presentation Number")
            if info_div:
                property_div = info_div.find_next_sibling("div", class_="property")
                if property_div:
                    number = property_div.text.strip()
        abstract_number = number
        title = card.find('h4', class_='card-title').text.strip()
        # Find the speakers
        speakers_container = card.find("div", class_="property-container")
        speakers = "N/A"
        if speakers_container:
            info_div = speakers_container.find("div", class_="info", string="Speakers")
            if info_div:
                property_div = info_div.find_next_sibling("div", class_="property")
                if property_div:
                    speakers = property_div.text.strip()

        # Use speakers for authors
        authors = speakers
        # Find the presentation date
        date_container = card.find('div', class_='property-container cslide text p')
        presentation_date = "N/A"
        if date_container:
            property_div = date_container.find('div', class_='property property-text')
            if property_div:
                presentation_date = property_div.text.strip()
        # Find the topic
        topic_element = card.find('span', class_='label p a-pt a-pt-hover')
        topic = topic_element.text.strip() if topic_element else "N/A"        
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

print("CSV file 'abstracts_2.csv' has been created successfully.")
