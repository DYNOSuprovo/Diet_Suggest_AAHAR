

// lib/screens/profile_screen.dart (New Screen)
import 'package:flutter/material.dart';
import 'package:my_health_app/utils/app_constants.dart';

class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key});

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
        title: const Text('My Profile'),
        backgroundColor: AppConstants.primaryColor,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            Container(
              color: AppConstants.primaryColor,
              padding: const EdgeInsets.symmetric(vertical: 24.0, horizontal: 16.0),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      _WeightDisplay(
                        label: 'Current Weight',
                        value: '61.0 Kg',
                      ),
                      _WeightDisplay(
                        label: 'Target Weight',
                        value: '72.0 Kg',
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),
                  const CircleAvatar(
                    radius: 40,
                    backgroundColor: Colors.white,
                    child: Text('R', style: TextStyle(fontSize: 40, color: AppConstants.primaryColor)), // Placeholder for user initial/image
                  ),
                  const SizedBox(height: 10),
                  const Text(
                    'Rish mish',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const Text(
                    'rishm35@gmail.com',
                    style: TextStyle(
                      fontSize: 14,
                      color: Color(0xFFE8F5E9), // Lighter green
                    ),
                  ),
                  const Text(
                    'User since May 22 2025',
                    style: TextStyle(
                      fontSize: 12,
                      color: Color(0xFFE8F5E9),
                    ),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                children: [
                  _ProfileInfoTile(
                    icon: Icons.info_outline,
                    title: 'Basic Information',
                    subtitle: 'Height, Weight, Age, Gender, BMI',
                    onTap: () {
                      // TODO: Navigate to basic info edit screen
                    },
                  ),
                  _ProfileInfoTile(
                    icon: Icons.track_changes,
                    title: 'Goal',
                    subtitle: 'Weight gain',
                    onTap: () {
                      // TODO: Navigate to goal edit screen
                    },
                  ),
                  _ProfileInfoTile(
                    icon: Icons.location_on_outlined,
                    title: 'Location',
                    subtitle: 'IN, India',
                    trailing: const Text('Visible only to you', style: TextStyle(color: AppConstants.greyColor)),
                    onTap: () {
                      // TODO: Navigate to location edit screen
                    },
                  ),
                  _ProfileInfoTile(
                    icon: Icons.food_bank_outlined,
                    title: 'Food Preferences',
                    subtitle: 'Food Preferences, Cuisine, Meal',
                    onTap: () {
                      // TODO: Navigate to food preferences edit screen
                    },
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

class _WeightDisplay extends StatelessWidget {
  final String label;
  final String value;

  const _WeightDisplay({
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          label,
          style: const TextStyle(
            color: Colors.white70,
            fontSize: 12,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }
}

class _ProfileInfoTile extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final Widget? trailing;
  final VoidCallback? onTap;

  const _ProfileInfoTile({
    required this.icon,
    required this.title,
    required this.subtitle,
    this.trailing,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8.0),
      elevation: 0, // No elevation for these cards as per image
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(0)), // No rounded corners
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 12.0, horizontal: 0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(icon, color: AppConstants.textColor),
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
                  if (trailing != null) trailing!,
                  const Icon(Icons.arrow_forward_ios, size: 18, color: AppConstants.greyColor),
                ],
              ),
              const Divider(height: 24, thickness: 1, color: AppConstants.lightGrey), // Divider line
            ],
          ),
        ),
      ),
    );
  }
}

