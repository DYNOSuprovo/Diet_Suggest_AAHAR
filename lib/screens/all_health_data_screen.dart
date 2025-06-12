
// lib/screens/all_health_data_screen.dart (New Screen)
import 'package:flutter/material.dart';
import 'package:my_health_app/utils/app_constants.dart';

class AllHealthDataScreen extends StatelessWidget {
  const AllHealthDataScreen({super.key});

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
        title: const Text('All Health Data'),
        backgroundColor: Colors.transparent, // Make AppBar transparent to match design
        foregroundColor: AppConstants.textColor,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            _HealthDataItem(
              icon: Icons.access_time_rounded,
              color: AppConstants.primaryColor,
              title: 'Double Support Time',
              value: '29.7 %',
            ),
            _HealthDataItem(
              icon: Icons.directions_walk,
              color: const Color(0xFFFFA726), // Orange
              title: 'Steps',
              value: '11,875 steps',
            ),
            _HealthDataItem(
              icon: Icons.calendar_today,
              color: AppConstants.primaryColor,
              title: 'Cycle tracking',
              value: '08 April',
            ),
            _HealthDataItem(
              icon: Icons.bed,
              color: const Color(0xFF66BB6A), // Green
              title: 'Sleep',
              value: '7 hr 31 min',
            ),
            _HealthDataItem(
              icon: Icons.favorite,
              color: Colors.red,
              title: 'Heart',
              value: '68 BPM',
            ),
            _HealthDataItem(
              icon: Icons.local_fire_department,
              color: AppConstants.accentColor, // Lighter green
              title: 'Burned calories',
              value: '850 kcal',
            ),
            _HealthDataItem(
              icon: Icons.scale, // Placeholder icon for BMI
              color: AppConstants.primaryColor,
              title: 'Body mass index',
              value: '18,69 BMI',
            ),
          ],
        ),
      ),
    );
  }
}

class _HealthDataItem extends StatelessWidget {
  final IconData icon;
  final Color color;
  final String title;
  final String value;

  const _HealthDataItem({
    required this.icon,
    required this.color,
    required this.title,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8.0),
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: color.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Icon(icon, color: color, size: 28),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                      color: AppConstants.textColor,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    value,
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: AppConstants.textColor,
                    ),
                  ),
                ],
              ),
            ),
            const Icon(Icons.arrow_forward_ios, size: 18, color: AppConstants.greyColor),
          ],
        ),
      ),
    );
  }
}
