// Native fetch

const keys = [
    //    "AIzaSyDJsNRo0nqIgVJ448lAssED1UrKdD-T5pc",
    "AIzaSyDba6Xiaff6F8kqsxUVvslbBU0bzayV_Z8"
];

async function testKey(key, index) {
    console.log(`Testing Key ${index + 1}: ${key.substring(0, 10)}...`);
    try {
        const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${key}`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (response.ok) {
            console.log(`✅ Key ${index + 1} is WORKING.`);
            if (data.models) {
                console.log(`   Available Models: ${data.models.map(m => m.name.replace('models/', '')).slice(0, 5).join(', ')}...`);
            }
        } else {
            console.log(`❌ Key ${index + 1} FAILED.`);
            console.log("Error Detail:", JSON.stringify(data.error, null, 2));
        }
    } catch (error) {
        console.error(`❌ Key ${index + 1} Error:`, error.message);
    }
    console.log("------------------------------------------------");
}

async function run() {
    // using native fetch if node-fetch fails, so removing require if needed in future steps but standard node environment usually needs it or flag.
    // Actually Node 18+ has native fetch. I'll remove require.
    for (let i = 0; i < keys.length; i++) {
        await testKey(keys[i], i);
    }
}

run();
