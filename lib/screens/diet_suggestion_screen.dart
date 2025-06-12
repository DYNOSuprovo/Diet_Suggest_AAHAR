


// lib/screens/diet_suggestion_screen.dart (New Screen)
import 'package:flutter/material.dart';
import 'package:my_health_app/utils/app_constants.dart';

class DietSuggestionScreen extends StatelessWidget {
  const DietSuggestionScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
        title: const Text('AI-Powered Diet Suggestion'),
        backgroundColor: Colors.transparent,
        foregroundColor: AppConstants.textColor,
        actions: [
          TextButton(
            onPressed: () {
              // TODO: Implement upgrade action
            },
            child: const Text(
              'Upgrade',
              style: TextStyle(color: AppConstants.primaryColor, fontWeight: FontWeight.bold),
            ),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Maintain your current weight by focusing on foods that balance your hemoglobin, glucose, and cholesterol levels.',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    color: AppConstants.textColor,
                  ),
            ),
            const SizedBox(height: 30),
            Center(
              child: Image.network(
                'https://placehold.co/200x150/E0E0E0/000000?text=Food+Icons', // Placeholder for broccoli and salmon
                height: 150,
                width: 200,
                fit: BoxFit.contain,
              ),
            ),
            const SizedBox(height: 30),
            _DietOptionCard(
              type: 'Free',
              title: 'AI Diet Suggestions',
              description: 'Customized diet recommendations, free of charge',
              icon: Icons.check_circle_outline, // Placeholder icon
              iconColor: AppConstants.primaryColor,
            ),
            const SizedBox(height: 20),
            _DietOptionCard(
              type: 'Premium',
              title: 'Hyperlocal Ingredients',
              description: 'Personalized recipes using ingredients sourced near you',
              icon: Icons.star_border, // Placeholder icon
              iconColor: Colors.amber,
            ),
            const SizedBox(height: 30),
            Row(
              children: [
                const Icon(Icons.headset_mic_outlined, color: AppConstants.greyColor),
                const SizedBox(width: 8),
                Text(
                  '3 free trials available this month',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: AppConstants.greyColor,
                      ),
                ),
              ],
            ),
            const SizedBox(height: 30),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () {
                  // TODO: Continue logic
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppConstants.primaryColor,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16.0),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12.0),
                  ),
                ),
                child: const Text(
                  'Continue',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _DietOptionCard extends StatelessWidget {
  final String type;
  final String title;
  final String description;
  final IconData icon;
  final Color iconColor;

  const _DietOptionCard({
    required this.type,
    required this.title,
    required this.description,
    required this.icon,
    required this.iconColor,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, size: 36, color: iconColor),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: iconColor.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          type,
                          style: TextStyle(
                            color: iconColor,
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        title,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: AppConstants.textColor,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    description,
                    style: const TextStyle(
                      fontSize: 14,
                      color: AppConstants.greyColor,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
