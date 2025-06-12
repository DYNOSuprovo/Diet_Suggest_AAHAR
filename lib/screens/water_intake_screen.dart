

// lib/screens/water_intake_screen.dart (New Screen)
import 'package:flutter/material.dart';
import 'package:my_health_app/utils/app_constants.dart';

class WaterIntakeScreen extends StatelessWidget {
  const WaterIntakeScreen({super.key});

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
        title: const Text('Water Intake'),
        backgroundColor: Colors.transparent,
        foregroundColor: AppConstants.textColor,
        actions: [
          IconButton(
            icon: const Icon(Icons.notifications_none),
            onPressed: () {
              // TODO: Handle notifications
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Stack(
                alignment: Alignment.center,
                children: [
                  SizedBox(
                    width: 200,
                    height: 200,
                    child: CircularProgressIndicator(
                      value: 1500 / 2500, // 1500ml out of 2500ml goal
                      strokeWidth: 15,
                      backgroundColor: AppConstants.lightGrey,
                      color: AppConstants.darkBlue,
                    ),
                  ),
                  Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        '1500ml',
                        style: Theme.of(context).textTheme.displayLarge?.copyWith(
                              fontWeight: FontWeight.bold,
                              color: AppConstants.darkBlue,
                            ),
                      ),
                      Text(
                        'of 2500ml',
                        style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                              color: AppConstants.greyColor,
                            ),
                      ),
                      const SizedBox(height: 10),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          _WaterActionButton(
                            icon: Icons.remove,
                            onPressed: () {
                              // TODO: Decrease water intake
                            },
                          ),
                          const SizedBox(width: 20),
                          _WaterActionButton(
                            icon: Icons.add,
                            onPressed: () {
                              // TODO: Increase water intake
                            },
                          ),
                        ],
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 30),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppConstants.darkBlue.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                children: [
                  const Icon(Icons.water_drop, color: AppConstants.darkBlue, size: 28),
                  const SizedBox(width: 16),
                  Text(
                    'You are 60 % hydrated !',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: AppConstants.darkBlue,
                        ),
                  ),
                  const Spacer(),
                  const Icon(Icons.info_outline, color: AppConstants.darkBlue),
                ],
              ),
            ),
            const SizedBox(height: 30),
            Text(
              'Hydration History',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
                color: AppConstants.textColor,
              ),
            ),
            const SizedBox(height: 15),
            _HydrationHistoryItem(
              time: '10:30 AM',
              amount: '250ml',
              source: 'Water',
              sourceColor: AppConstants.darkBlue,
            ),
            _HydrationHistoryItem(
              time: '12:45 PM',
              amount: '500ml',
              source: 'Smoothie',
              sourceColor: AppConstants.primaryColor,
            ),
            _HydrationHistoryItem(
              time: '03:15 PM',
              amount: '250ml',
              source: 'Coffee',
              sourceColor: Colors.brown,
            ),
            _HydrationHistoryItem(
              time: '06:00 PM',
              amount: '500ml',
              source: 'Water',
              sourceColor: AppConstants.darkBlue,
            ),
            const SizedBox(height: 30),
            Text(
              'Reminders',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
                color: AppConstants.textColor,
              ),
            ),
            const SizedBox(height: 15),
            _ReminderTile(
              title: 'Reminders Active',
              subtitle: 'Every 2 hours',
              onTap: () {
                // TODO: Navigate to reminder settings
              },
            ),
          ],
        ),
      ),
    );
  }
}

class _WaterActionButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onPressed;

  const _WaterActionButton({
    required this.icon,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onPressed,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: AppConstants.darkBlue,
          shape: BoxShape.circle,
        ),
        child: Icon(icon, color: Colors.white, size: 24),
      ),
    );
  }
}

class _HydrationHistoryItem extends StatelessWidget {
  final String time;
  final String amount;
  final String source;
  final Color sourceColor;

  const _HydrationHistoryItem({
    required this.time,
    required this.amount,
    required this.source,
    required this.sourceColor,
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
            const Icon(Icons.timer_outlined, color: AppConstants.greyColor, size: 24),
            const SizedBox(width: 16),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  time,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w500,
                    color: AppConstants.textColor,
                  ),
                ),
                Text(
                  amount,
                  style: const TextStyle(
                    fontSize: 14,
                    color: AppConstants.greyColor,
                  ),
                ),
              ],
            ),
            const Spacer(),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              decoration: BoxDecoration(
                color: sourceColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                source,
                style: TextStyle(
                  color: sourceColor,
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ReminderTile extends StatelessWidget {
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const _ReminderTile({
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8.0),
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Row(
            children: [
              const Icon(Icons.notifications_none, color: AppConstants.primaryColor, size: 28),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: AppConstants.textColor,
                      ),
                    ),
                    Text(
                      subtitle,
                      style: const TextStyle(
                        fontSize: 14,
                        color: AppConstants.greyColor,
                      ),
                    ),
                  ],
                ),
              ),
              const Icon(Icons.arrow_forward, color: AppConstants.greyColor),
            ],
          ),
        ),
      ),
    );
  }
}

