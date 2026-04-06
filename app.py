from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load AI models
energy_model = joblib.load('energy_model.joblib')
co2_model = joblib.load('co2_model.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        people = float(request.form.get('people'))
        ac_hours = float(request.form.get('ac_hours'))
        bill = float(request.form.get('bill'))
        city = request.form.get('city')

        # AI prediction
        energy = energy_model.predict([[people, ac_hours, bill]])[0]
        co2 = co2_model.predict([[people, ac_hours, bill]])[0]

        # For chart: compare with average Indian household
        avg_energy = 350

        return f"""
        <h2 style="color:green; text-align:center;">✅ AI Prediction Result</h2>
        <p style="text-align:center; font-size:22px;">
            For a house with <b>{people}</b> people in <b>{city}</b> using AC <b>{ac_hours}</b> hours:<br><br>
            📊 Predicted monthly energy: <b>{energy:.0f} kWh</b><br>
            🌍 Carbon footprint: <b>{co2:.0f} kg CO₂</b><br><br>
            💡 Suggestion: Turn off AC 2 hours early to save around 80 kWh!
        </p>

        <!-- Chart -->
        <div style="max-width:600px; margin:30px auto;">
            <canvas id="energyChart" width="600" height="300"></canvas>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            const ctx = document.getElementById('energyChart');
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: ['Your House', 'Average House'],
                    datasets: [{{
                        label: 'Monthly Energy (kWh)',
                        data: [{energy:.0f}, {avg_energy}],
                        backgroundColor: ['#2e8b57', '#a9a9a9']
                    }}]
                }},
                options: {{
                    scales: {{ y: {{ beginAtZero: true }} }}
                }}
            }});
        </script>

        <p style="text-align:center; margin-top:30px;">
            <a href="/" style="color:#2e8b57; font-size:18px;">🔄 Predict Again</a>
        </p>
        """
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)