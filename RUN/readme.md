
1. `streamlit_input.py` – to scrape or prepare data
2. `generate_index.py` – to process the data and generate an index
3. `streamlit_output.py` – to generate a video based on the indexed data

---

````markdown
# 📹 Streamlit Video Generator Workflow

This project provides a step-by-step workflow to scrape data, generate an index, and create a video using Streamlit apps and a backend indexing script.


1. Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Workflow

### Step 1: Run `streamlit_input.py` to Scrape/Prepare Data

Launch the Streamlit app to input or scrape data.

```bash
streamlit run streamlit_input.py
```

* This app allows you to gather or input the data needed for the next step.
* Once done, note the path where the data is saved (e.g., `data/scraped_data.json`).

---

### Step 2: Run `generate_index.py` to Process Data

Update the path of the scraped data in the script before running it.

#### 🔧 Edit `generate_index.py`:

Open the file and update the path variable like so:

```python
data_directory = "/home/sriya/StartCoding/startCoding_Video_Model/scraped_data"  # Update this path if needed
```

Then run the script:

```bash
python3 generate_index.py
```

* This script will read the scraped data and generate an index used for generating the video.

---

### Step 3: Run `streamlit_output.py` to Generate the Video

Launch the Streamlit app for video generation:

```bash
streamlit run streamlit_output.py
```

* This app will use the indexed data to generate the final video output.
* Make sure the index file exists before running this app.

---

## 🧾 Notes

* Ensure all files are correctly saved and paths are valid between steps.
* You can customize the data format and index generation logic as per your requirements.

---
