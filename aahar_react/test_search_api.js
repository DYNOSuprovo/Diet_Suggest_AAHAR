const BASE_URL = "https://diet-suggest-aahar.onrender.com";

async function testSearch(term) {
    try {
        console.log(`Searching for ${term}...`);
        const response = await fetch(`${BASE_URL}/nutrition/search/${term}`);

        console.log("Status:", response.status);
        if (response.ok) {
            const data = await response.json();
            console.log("Data structure:", JSON.stringify(data).substring(0, 500));
            console.log("Keys:", Object.keys(data));
        } else {
            const text = await response.text();
            console.log("Error:", text);
        }
    } catch (error) {
        console.error("Fetch error:", error);
    }
}

testSearch("sheermal");
