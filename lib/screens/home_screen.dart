
// lib/screens/home_screen.dart (Significantly Modified)
import 'package:flutter/material.dart';
import 'package:my_health_app/utils/app_constants.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0; // For bottom navigation

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
    // Implement navigation logic for bottom bar
    switch (index) {
      case 0:
      // Already on home
        break;
      case 1:
        Navigator.pushNamed(context, '/diet_plan'); // Assuming Plans tab goes to Diet Plan
        break;
      case 2:
        Navigator.pushNamed(context, '/meal_tracker'); // Assuming Log tab goes to Meal Tracker
        break;
      case 3:
        Navigator.pushNamed(context, '/profile');
        break;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: Padding(
          padding: const EdgeInsets.all(8.0),
          child: Image.network(
            'https://placehold.co/40x40/FFFFFF/4CAF50?text=Aaliar', // Placeholder for "Aaliar" logo
            height: 40,
            width: 40,
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.person), // Placeholder for user profile image
            onPressed: () {
              Navigator.pushNamed(context, '/profile');
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Hi User',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
                color: AppConstants.textColor,
              ),
            ),
            const SizedBox(height: 20),
            // Calendar Section
            Container(
              padding: const EdgeInsets.all(16.0),
              decoration: BoxDecoration(
                color: AppConstants.lightGrey,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      IconButton(onPressed: () {}, icon: const Icon(Icons.arrow_back_ios, size: 16)),
                      Text(
                        'June 2025',
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
                      ),
                      IconButton(onPressed: () {}, icon: const Icon(Icons.arrow_forward_ios, size: 16)),
                    ],
                  ),
                  const SizedBox(height: 10),
                  const Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      Text('Sun', style: TextStyle(color: AppConstants.greyColor)),
                      Text('Mon', style: TextStyle(color: AppConstants.greyColor)),
                      Text('Tue', style: TextStyle(color: AppConstants.greyColor)),
                      Text('Wed', style: TextStyle(color: AppConstants.greyColor)),
                      Text('Thu', style: TextStyle(color: AppConstants.greyColor)),
                      Text('Fri', style: TextStyle(color: AppConstants.greyColor)),
                      Text('Sat', style: TextStyle(color: AppConstants.greyColor)),
                    ],
                  ),
                  const SizedBox(height: 10),
                  GridView.builder(
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 7,
                      childAspectRatio: 1.0,
                      mainAxisSpacing: 8,
                      crossAxisSpacing: 8,
                    ),
                    itemCount: 7, // Just showing a week for simplicity
                    itemBuilder: (context, index) {
                      final day = index + 1;
                      final isSelected = day == 2; // Example: Monday 2nd is selected
                      return GestureDetector(
                        onTap: () {
                          // TODO: Handle date selection
                        },
                        child: Container(
                          decoration: BoxDecoration(
                            color: isSelected ? AppConstants.primaryColor : Colors.transparent,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          alignment: Alignment.center,
                          child: Text(
                            '$day',
                            style: TextStyle(
                              color: isSelected ? Colors.white : AppConstants.textColor,
                              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),
            const SizedBox(height: 30),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Daily Overview',
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: AppConstants.textColor,
                  ),
                ),
                TextButton(
                  onPressed: () {
                    Navigator.pushNamed(context, '/all_health_data');
                  },
                  child: const Text(
                    'View more >',
                    style: TextStyle(color: AppConstants.primaryColor, fontWeight: FontWeight.w600),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 15),
            GridView.count(
              crossAxisCount: 2,
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              mainAxisSpacing: 16.0,
              crossAxisSpacing: 16.0,
              children: const [
                _OverviewCard(
                  icon: Icons.hotel_rounded,
                  title: 'Sleep',
                  value: '8 hours',
                  color: Color(0xFFBBDEFB), // Light Blue
                ),
                _OverviewCard(
                  icon: Icons.water_drop,
                  title: 'Hydration',
                  value: '1,200 ml',
                  color: Color(0xFFC8E6C9), // Light Green
                ),
                _OverviewCard(
                  icon: Icons.directions_run,
                  title: 'Exercise',
                  value: '1 hour',
                  color: Color(0xFFFFCC80), // Orange
                ),
                _OverviewCard(
                  icon: Icons.fastfood,
                  title: 'Nutrition',
                  value: '1,500 kcal',
                  color: Color(0xFFD1C4E9), // Light Purple
                ),
              ],
            ),
            const SizedBox(height: 30),
            // Achieve Daily Goals Card
            GestureDetector(
              onTap: () {
                Navigator.pushNamed(context, '/diet_suggestion'); // Example navigation
              },
              child: Container(
                padding: const EdgeInsets.all(16.0),
                decoration: BoxDecoration(
                  color: AppConstants.primaryColor,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Achieve Your Daily Goals !',
                            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                ),
                          ),
                          const SizedBox(height: 5),
                          Text(
                            'Tap to view your personalized progress and targets',
                            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                  color: Colors.white.withOpacity(0.8),
                                ),
                          ),
                        ],
                      ),
                    ),
                    const Icon(Icons.arrow_forward_ios, color: Colors.white, size: 20),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 30),
            Text(
              'Quick Actions',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
                color: AppConstants.textColor,
              ),
            ),
            const SizedBox(height: 15),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _QuickActionButton(
                  icon: Icons.restaurant_menu,
                  label: 'Diet Plan',
                  onPressed: () {
                    Navigator.pushNamed(context, '/diet_plan');
                  },
                ),
                _QuickActionButton(
                  icon: Icons.local_dining,
                  label: 'Meal Log',
                  onPressed: () {
                    Navigator.pushNamed(context, '/meal_tracker');
                  },
                ),
                _QuickActionButton(
                  icon: Icons.water_drop_outlined,
                  label: 'Water Track',
                  onPressed: () {
                    Navigator.pushNamed(context, '/water_intake');
                  },
                ),
                _QuickActionButton(
                  icon: Icons.fitness_center_outlined, // Placeholder, might need to change icon
                  label: 'Fitness Sync',
                  onPressed: () {
                    Navigator.pushNamed(context, '/fitness_sync');
                  },
                ),
              ],
            ),
            const SizedBox(height: 30),
            Text(
              'Recommended Meals',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
                color: AppConstants.textColor,
              ),
            ),
            const SizedBox(height: 15),
            SizedBox(
              height: 200, // Adjust height as needed
              child: ListView(
                scrollDirection: Axis.horizontal,
                children: const [
                  _RecommendedMealCard(
                    image: 'https://placehold.co/150x150/E0E0E0/000000?text=Smoothie',
                    title: 'Super Green Smoothie Bowl',
                    description: 'A refreshing and nutritious blend of...',
                  ),
                  SizedBox(width: 16),
                  _RecommendedMealCard(
                    image: 'https://placehold.co/150x150/E0E0E0/000000?text=Salad',
                    title: 'Quinoa Salad with Roasted Veggies',
                    description: 'A hearty and wholesome salad featuring fluffy...',
                  ),
                  SizedBox(width: 16),
                  _RecommendedMealCard(
                    image: 'https://placehold.co/150x150/E0E0E0/000000?text=Dish',
                    title: 'Chicken & Veggie Stir-fry',
                    description: 'A quick and easy stir-fry with lean protein and colorful vegetables.',
                  ),
                ],
              ),
            ),
            const SizedBox(height: 30),
            Text(
              'Nutritional Summary',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
                color: AppConstants.textColor,
              ),
            ),
            const SizedBox(height: 15),
            Center(
              child: SizedBox(
                width: 200,
                height: 200,
                child: CustomPaint(
                  painter: _PieChartPainter(), // Custom painter for the pie chart
                ),
              ),
            ),
            const SizedBox(height: 20),
            const _NutritionalLegend(
              carbohydrates: '400g',
              proteins: '250g',
              fats: '150g',
              other: '50g',
            ),
            const SizedBox(height: 20), // For bottom padding
          ],
        ),
      ),
      bottomNavigationBar: BottomNavigationBar(
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.home_outlined),
            label: 'Home',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.article_outlined), // Changed icon for Plans
            label: 'Plans',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.assignment_outlined), // Changed icon for Log
            label: 'Log',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person_outline),
            label: 'Profile',
          ),
        ],
        currentIndex: _selectedIndex,
        selectedItemColor: AppConstants.primaryColor,
        unselectedItemColor: AppConstants.greyColor,
        onTap: _onItemTapped,
        type: BottomNavigationBarType.fixed,
        backgroundColor: Colors.white,
        elevation: 10,
        selectedLabelStyle: const TextStyle(fontWeight: FontWeight.w600),
        unselectedLabelStyle: const TextStyle(fontWeight: FontWeight.w500),
      ),
    );
  }
}

class _OverviewCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String value;
  final Color color;

  const _OverviewCard({
    required this.icon,
    required this.title,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12.0),
      ),
      color: color.withOpacity(0.2), // Lighter shade of the card color
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Align(
              alignment: Alignment.topRight,
              child: Icon(
                icon,
                size: 30,
                color: color,
              ),
            ),
            const SizedBox(height: 10),
            Text(
              title,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w500,
                color: AppConstants.textColor,
              ),
            ),
            Text(
              value,
              style: const TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: AppConstants.textColor,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _QuickActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onPressed;

  const _QuickActionButton({
    required this.icon,
    required this.label,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onPressed,
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: AppConstants.lightGrey,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, size: 30, color: AppConstants.primaryColor),
          ),
          const SizedBox(height: 8),
          Text(
            label,
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w500,
              color: AppConstants.textColor,
            ),
          ),
        ],
      ),
    );
  }
}

class _RecommendedMealCard extends StatelessWidget {
  final String image;
  final String title;
  final String description;

  const _RecommendedMealCard({
    required this.image,
    required this.title,
    required this.description,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: SizedBox(
        width: 180,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ClipRRect(
              borderRadius: const BorderRadius.vertical(top: Radius.circular(12)),
              child: Image.network(
                image,
                height: 100,
                width: double.infinity,
                fit: BoxFit.cover,
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 14,
                      color: AppConstants.textColor,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    description,
                    style: const TextStyle(
                      fontSize: 12,
                      color: AppConstants.greyColor,
                    ),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
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

class _PieChartPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.fill;
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width / 2;

    // Data for the pie chart segments (values in percentages for visual representation)
    // Carbohydrates (400g)
    // Proteins (250g)
    // Fats (150g)
    // Other (50g)
    // Total = 400+250+150+50 = 850
    // Carb: 400/850 = 0.47 (approx) -> 47%
    // Proteins: 250/850 = 0.29 (approx) -> 29%
    // Fats: 150/850 = 0.17 (approx) -> 17%
    // Other: 50/850 = 0.05 (approx) -> 5%
    final List<double> percentages = [0.47, 0.29, 0.17, 0.05];
    final List<Color> colors = [
      AppConstants.primaryColor,
      AppConstants.accentColor,
      AppConstants.darkBlue,
      AppConstants.greyColor,
    ];

    double startAngle = -0.5 * 3.14159; // Start from the top (12 o'clock)

    for (int i = 0; i < percentages.length; i++) {
      final sweepAngle = percentages[i] * 2 * 3.14159;
      paint.color = colors[i];
      canvas.drawArc(
        Rect.fromCircle(center: center, radius: radius),
        startAngle,
        sweepAngle,
        true, // Use center for filled arcs
        paint,
      );
      startAngle += sweepAngle;
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return false;
  }
}

class _NutritionalLegend extends StatelessWidget {
  final String carbohydrates;
  final String proteins;
  final String fats;
  final String other;

  const _NutritionalLegend({
    required this.carbohydrates,
    required this.proteins,
    required this.fats,
    required this.other,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _LegendItem(
          color: AppConstants.primaryColor,
          text: 'Carbohydrates ($carbohydrates)',
        ),
        _LegendItem(
          color: AppConstants.accentColor,
          text: 'Proteins ($proteins)',
        ),
        _LegendItem(
          color: AppConstants.darkBlue,
          text: 'Fats ($fats)',
        ),
        _LegendItem(
          color: AppConstants.greyColor,
          text: 'Other ($other)',
        ),
      ],
    );
  }
}

class _LegendItem extends StatelessWidget {
  final Color color;
  final String text;

  const _LegendItem({
    required this.color,
    required this.text,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        children: [
          Container(
            width: 16,
            height: 16,
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(4),
            ),
          ),
          const SizedBox(width: 8),
          Text(
            text,
            style: const TextStyle(fontSize: 14, color: AppConstants.textColor),
          ),
        ],
      ),
    );
  }
}
