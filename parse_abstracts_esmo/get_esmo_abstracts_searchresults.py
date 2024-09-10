import csv
from bs4 import BeautifulSoup

# Read the HTML file
with open("data/ESMO Congress 2024 - Conference Calendar - ESMO Congress 2024.html", "r") as file:
    html_content = file.read()

# Create a BeautifulSoup object
soup = BeautifulSoup(html_content, "html.parser")

# Find all the abstract divs
abstract_divs = soup.find_all("div", class_="abstract")
# Open a CSV file for writing
with open("abstracts.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Title", "Presentation Number", "Speakers", "Abstract"])

    for abstract_div in abstract_divs:
        # Find the title
        title_element = abstract_div.find_previous("h4", class_="card-title")
        title = title_element.text.strip() if title_element else "N/A"

        # Find the presentation number
        number_container = abstract_div.find_previous("div", class_="property-container")
        number = "N/A"
        if number_container:
            info_div = number_container.find("div", class_="info", string="Presentation Number")
            if info_div:
                property_div = info_div.find_next_sibling("div", class_="property")
                if property_div:
                    number = property_div.text.strip()

        # Find the speakers
        speakers_container = abstract_div.find_previous("div", class_="property-container")
        speakers = "N/A"
        if speakers_container:
            info_div = speakers_container.find("div", class_="info", string="Speakers")
            if info_div:
                property_div = info_div.find_next_sibling("div", class_="property")
                if property_div:
                    speakers = property_div.text.strip()

        # Extract the abstract text
        abstract_text = abstract_div.text.strip()

        # Write the extracted information to the CSV file
        writer.writerow([title, number, speakers, abstract_text])

print("Extraction complete. Results saved to abstracts.csv")