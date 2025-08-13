import requests
# Your OpenWeatherMap API key
API_KEY = "cbf74aade9b233c5dc5f99b1b49f7d50"  
def get_weather(city):
    """Fetch current weather data for a given city."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            print(f"Error: {data.get('message', 'Unable to fetch weather')}")
            return None

        temp = data['main']['temp']
        weather_condition = data['weather'][0]['description']
        humidity = data['main']['humidity']

        return {
            "city": city,
            "temperature": temp,
            "condition": weather_condition,
            "humidity": humidity
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
if __name__ == "__main__":
    city = "Kolkata"  # Default city
    weather_data = get_weather(city)
    if weather_data:
        print(f"\nWeather in {weather_data['city']}:")
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Condition: {weather_data['condition']}")
        print(f"Humidity: {weather_data['humidity']}%")
