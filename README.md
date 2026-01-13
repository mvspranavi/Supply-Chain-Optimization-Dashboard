#  Supply Chain Optimization Dashboard

An interactive **Streamlit-based web application** for visualizing, analyzing, and optimizing end-to-end supply chain networks.  
It integrates data from multiple CSV sources (suppliers, plants, warehouses, customers, and transport routes) and uses AI-generated insights to suggest cost-saving strategies and performance improvements.

##  Features

- 📊 **Dynamic Data Upload:** Import and update CSV datasets for each supply chain entity.
- 🔄 **Real-Time Cost Recalculation:** Any change in data instantly updates overall cost metrics.
- 🧠 **AI-Driven Insights:** Built-in assistant provides natural language explanations, optimization hints, and sensitivity analysis.
- 🗺️ **Network Visualization:** Displays the flow from supplier → plant → warehouse → customer.
- ⚙️ **Flexible Configuration:** Fully customizable structure for new datasets and parameters.


## 🗂️ Project Structure

📦 supply-chain-optimizer/  
├── app.py  
├── data/  
│ ├── suppliers.csv  
│ ├── plants.csv  
│ ├── warehouses.csv  
│ ├── customers.csv  
│ ├── transport_costs.csv  
├── requirements.txt  
├── .gitignore  
├── .env # contains credentials (not tracked in Git)  
└── README.md  


## ⚡️ How It Works

1. Upload or edit CSV files inside the `data/` folder.  
2. The app automatically reads data and builds the supply chain network.  
3. It computes total operational cost and identifies bottlenecks.  
4. The AI assistant generates explanations and suggestions in Markdown format.  

---

## Example CSV Files

### suppliers.csv
```csv
supplier_id,location,capacity,cost_per_unit
S1,LOC_A,1000,3.5
S2,LOC_B,700,3.8
transport_costs.csv
from_id,to_id,cost_per_unit,lead_time_days
S1,P1,0.5,1
P1,W1,0.8,2
W1,C1,1.5,3
```
Installation
1.Clone this repository:
```csv
git clone https://github.com/<your-username>/supply-chain-optimizer.git
cd supply-chain-optimizer
```

2.Create and activate a virtual environment:
```csv
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

3.Install dependencies:
```csv
pip install -r requirements.txt
```

4.Run the app:
```csv
streamlit run app.py
```

Environment Variables  
Create a .env file in the root directory to store private keys or credentials:  
OPENAI_API_KEY=your_key_here  
.env is excluded via .gitignore for security.  
Example Prompts You Can Ask the AI are in prompts.txt  

Future Enhancements  
Add optimization solver (e.g., PuLP / OR-Tools)  
Integrate real-time logistics tracking API  
Export network visualization as image or PDF  
Add Monte Carlo simulation for risk analysis  

Author  
Muskan Sohaney  
B.Tech CSE (AI & Analysis) | Data Analyst | AI/ML Enthusiast  

