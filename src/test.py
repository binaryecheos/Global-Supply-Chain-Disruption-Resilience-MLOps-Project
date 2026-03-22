import json
import requests

url = "https://gsc-test-926f.onrender.com/predict"



input = {
  "origin_city": "Mumbai, IN",
  "destination_city": "Los Angeles, US",
  "route_type": "Atlantic",
  "transportation_mode": "Air",
  "product_category": "Auto Parts",
  "delivery_status": "On Time",
  "disruption_event": "No Disruption",
  "base_lead_time_days": 35,
  "scheduled_lead_time_days": 38,
  "actual_lead_time_days": 35,
  "delay_days": 0,
  "geopolitical_risk_index": 0.73,
  "weather_severity_index": 8.3,
  "inflation_rate_pct": 4.31,
  "shipping_cost_usd": 4077.61,
  "order_weight_kg": 4714
}

response = requests.post(url, data=json.dumps(input) , headers={"Content-Type": "application/json"}) 
if response.status_code == 200:
    prediction = response.json()
    print("Predicted Mitigation Action:", prediction)
else:
    print("Error:", response.status_code)