from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time

options = Options()
options.add_argument("--headless")

driver = webdriver.Chrome(options=options)

# Example: Binance (Exchange)
url = "https://www.walletexplorer.com/wallet/Binance"
driver.get(url)
time.sleep(5)

soup = BeautifulSoup(driver.page_source, "html.parser")

rows = soup.find_all("tr")
data = []

for row in rows:
    cols = row.find_all("td")
    if len(cols) >= 2:
        link = cols[1].find("a")
        if link and "tx" in link.get("href", ""):
            tx_hash = link.get("href").split("/")[-1]
            date = cols[0].text.strip()
            label = "Exchange"   # entity-level label
            data.append([tx_hash, date, label])

df = pd.DataFrame(data, columns=["hash", "date", "label"])
df.to_csv("data/processed/walletexplorer_clean.csv", index=False)

driver.quit()

print("WalletExplorer cleaned data saved")
print("Shape:", df.shape)
