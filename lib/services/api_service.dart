import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "https://diet-aahar-fastapi.onrender.com/chat"; // Replace with your FastAPI IP:Port

  static Future<String> sendQuery(String query, String sessionId) async {
    try {
      final response = await http.post(
        Uri.parse(baseUrl),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "query": query,
          "session_id": sessionId,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data["answer"] ?? "No answer received.";
      } else {
        return "Error: ${response.statusCode}";
      }
    } catch (e) {
      return "Failed to connect to backend.";
    }
  }
}
